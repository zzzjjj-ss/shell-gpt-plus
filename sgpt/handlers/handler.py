import json
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional

from click.exceptions import UsageError

from ..cache import Cache
from ..config import cfg
from ..function import get_function
from ..printer import MarkdownPrinter, Printer, TextPrinter
from ..role import DefaultRoles, SystemRole

# ---------- 需要确认的函数与信任角色 ----------
try:
    FUNCTIONS_REQUIRE_CONFIRMATION = set(cfg.get("FUNCTIONS_REQUIRE_CONFIRMATION"))
except UsageError:
    FUNCTIONS_REQUIRE_CONFIRMATION = set()

try:
    TRUSTED_ROLES = set(cfg.get("TRUSTED_ROLES"))
except UsageError:
    TRUSTED_ROLES = set()

# ---------- API 调用方式（支持 LiteLLM 与原生 OpenAI，容错导入）----------
completion: Callable[..., Any] = lambda *args, **kwargs: Generator[Any, None, None]

base_url = cfg.get("API_BASE_URL")
use_litellm = cfg.get("USE_LITELLM") == "true"

base_request_kwargs = {
    "timeout": int(cfg.get("REQUEST_TIMEOUT")),
    "api_key": cfg.get("OPENAI_API_KEY"),
    "base_url": None if base_url == "default" else base_url,
}

if use_litellm:
    try:
        import litellm
        completion = litellm.completion
        litellm.suppress_debug_info = True
    except ImportError:
        from openai import OpenAI
        client = OpenAI(**base_request_kwargs)
        completion = client.chat.completions.create
        base_request_kwargs.clear()
else:
    from openai import OpenAI
    client = OpenAI(**base_request_kwargs)
    completion = client.chat.completions.create
    base_request_kwargs.clear()


# ---------- 公共清理函数（增强版）----------
def clean_tool_messages(messages: List[Dict]) -> List[Dict]:
    """
    移除所有不完整的工具调用消息：
    - 孤立的 tool 消息（前面没有 assistant 带 tool_calls）
    - 孤立的 assistant 带 tool_calls（后面缺少足够数量的 tool 消息）
    """
    cleaned = []
    i = 0
    n = len(messages)
    while i < n:
        msg = messages[i]
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            expected_tool_count = len(msg["tool_calls"])
            j = i + 1
            tool_count = 0
            while j < n and messages[j].get("role") == "tool":
                tool_count += 1
                j += 1
            if tool_count < expected_tool_count:
                # 不完整：跳过这个 assistant 及其后面跟随的 tool 消息
                i = j
                continue
            else:
                # 完整：保留 assistant 和所有 tool 消息
                cleaned.append(msg)
                cleaned.extend(messages[i+1:j])
                i = j
        elif msg.get("role") == "tool":
            # 孤立的 tool 消息（前面不是 assistant 带 tool_calls）
            i += 1
            continue
        else:
            cleaned.append(msg)
            i += 1
    return cleaned


class Handler:
    cache = Cache(int(cfg.get("CACHE_LENGTH")), Path(cfg.get("CACHE_PATH")))

    # ---------- 会话记忆存储目录 ----------
    SESSIONS_DIR = Path.home() / ".config" / "shell_gpt" / "sessions"
    _conversations: Dict[str, List[Dict[str, Any]]] = {}
    _current_session = "default"

    def __init__(
        self,
        role: SystemRole,
        markdown: bool,
        session_name: str = None,
    ) -> None:
        self.role = role
        api_base_url = cfg.get("API_BASE_URL")
        self.base_url = None if api_base_url == "default" else api_base_url
        self.timeout = int(cfg.get("REQUEST_TIMEOUT"))
        self.markdown = "APPLY MARKDOWN" in self.role.role and markdown
        self.code_theme = cfg.get("CODE_THEME")
        self.color = cfg.get("DEFAULT_COLOR")

        # 会话名：传入优先，否则用当前默认值
        self.session_name = session_name if session_name is not None else Handler._current_session
        Handler._current_session = self.session_name

        # 确保目录存在
        Handler.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def printer(self) -> Printer:
        if self.markdown:
            return MarkdownPrinter(self.code_theme)
        return TextPrinter(self.color)

    def _session_path(self) -> Path:
        safe_name = Path(self.session_name).name
        return Handler.SESSIONS_DIR / f"{safe_name}.json"

    def _load_session(self):
        if self.session_name in Handler._conversations:
            return
        file_path = self._session_path()
        if file_path.exists():
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    # 清理不完整的工具消息
                    data = clean_tool_messages(data)
                    Handler._conversations[self.session_name] = data
                    return
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        Handler._conversations[self.session_name] = []

    def _save_session(self):
        session = Handler._conversations.get(self.session_name)
        if session is None:
            return
        # 保存前也清理一次，防止写入垃圾数据
        cleaned_session = clean_tool_messages(session)
        file_path = self._session_path()
        file_path.write_text(
            json.dumps(cleaned_session, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def make_messages(self, prompt: str) -> List[Dict[str, Any]]:
        self._load_session()
        session = Handler._conversations.get(self.session_name)
        if not session:
            session.append({"role": "system", "content": self.role.role})
        session.append({"role": "user", "content": prompt})
        return session

    def _execute_tool_call(self, messages, tool_call):
        """执行工具调用，用 print 输出提示，不污染回复文本"""
        tool_call_id = tool_call["id"]
        name = tool_call["function"]["name"]
        arguments = tool_call["function"]["arguments"]
        dict_args = json.loads(arguments)
        joined_args = ", ".join(f'{k}="{v}"' for k, v in dict_args.items())
        print(f"> @FunctionCall `{name}({joined_args})` \n")

        execute_allowed = True
        if name in FUNCTIONS_REQUIRE_CONFIRMATION and self.role.name not in TRUSTED_ROLES:
            if name == "execute_shell_command":
                cmd = dict_args.get("shell_command", "")
                print(f"\n⚠️  About to execute: `{cmd}`")
            else:
                print(f"\n⚠️  About to call function `{name}` with args: {dict_args}")
            try:
                user_input = input("Proceed? (y/n): ").strip().lower()
            except EOFError:
                user_input = "n"
            execute_allowed = user_input == "y"
            if not execute_allowed:
                print("**User denied the function call.**\n")

        if execute_allowed:
            result = get_function(name)(**dict_args)
            if cfg.get("SHOW_FUNCTIONS_OUTPUT") == "true":
                print(f"```text\n{result}\n```\n")
        else:
            result = "User denied the function call."

        messages.append({"role": "tool", "content": result, "tool_call_id": tool_call_id})

    def get_completion(
        self,
        model: str,
        temperature: float,
        top_p: float,
        messages: List[Dict[str, Any]],
        functions: Optional[List[Dict[str, str]]],
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """
        流式生成器，负责输出助手回复的纯文本。
        移除了 @cache 装饰器，确保会话记录总是被更新。
        生成器正常结束时，会自动追加 assistant 消息并保存会话。
        """
        # 在开始前彻底清理消息序列，避免 API 400 错误
        messages = clean_tool_messages(messages)

        reasoning_content = ""
        tool_calls: List[dict] = []
        assistant_content = ""

        if self.role.name in {
            DefaultRoles.SHELL.value,
            DefaultRoles.CODE.value,
            DefaultRoles.DESCRIBE_SHELL.value,
        }:
            functions = None

        extra_kwargs = base_request_kwargs.copy()
        extra_kwargs.update(kwargs)
        if functions:
            extra_kwargs["tools"] = functions
            extra_kwargs["tool_choice"] = "auto"
            extra_kwargs["parallel_tool_calls"] = False

        response = completion(
            model=model,
            temperature=temperature,
            top_p=top_p,
            messages=messages,
            stream=True,
            **extra_kwargs,
        )

        try:
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    reasoning_content += delta.reasoning_content
                elif isinstance(delta, dict) and delta.get("reasoning_content"):
                    reasoning_content += delta["reasoning_content"]

                delta_tool_calls = delta.get("tool_calls") if use_litellm else delta.tool_calls
                if delta_tool_calls:
                    for tc in delta_tool_calls:
                        if use_litellm:
                            idx = tc.get("index", 0)
                            tool_id = tc.get("id") or ""
                            func = tc.get("function", {})
                            name = func.get("name", "")
                            args = func.get("arguments", "")
                        else:
                            idx = tc.index
                            tool_id = tc.id or ""
                            name = tc.function.name or ""
                            args = tc.function.arguments or ""

                        while len(tool_calls) <= idx:
                            tool_calls.append({"id": "", "function": {"name": "", "arguments": ""}})
                        if tool_id:
                            tool_calls[idx]["id"] = tool_id
                        if name:
                            tool_calls[idx]["function"]["name"] = name
                        tool_calls[idx]["function"]["arguments"] += args

                if delta.content:
                    assistant_content += delta.content
                    yield delta.content

                if chunk.choices[0].finish_reason == "tool_calls":
                    assistant_msg = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"id": tc["id"], "type": "function", "function": tc["function"]}
                            for tc in tool_calls
                        ],
                    }
                    if reasoning_content:
                        assistant_msg["reasoning_content"] = reasoning_content
                    messages.append(assistant_msg)

                    print()
                    for tc in tool_calls:
                        self._execute_tool_call(messages, tc)

                    # 递归继续，最终返回的文本会通过外层的 yield from 传出
                    yield from self.get_completion(
                        model=model,
                        temperature=temperature,
                        top_p=top_p,
                        messages=messages,
                        functions=functions,
                    )
                    return  # 递归结束后直接返回，不执行后续的保存

            # 正常结束（无工具调用或工具调用递归完成后）
            if assistant_content:
                assistant_msg = {"role": "assistant", "content": assistant_content}
                if reasoning_content:
                    assistant_msg["reasoning_content"] = reasoning_content
                messages.append(assistant_msg)

            # 不论是否产生内容，都将当前消息状态保存到磁盘
            self._save_session()

        except KeyboardInterrupt:
            response.close()
            raise

    def handle(
        self,
        prompt: str,
        model: str,
        temperature: float,
        top_p: float,
        caching: bool,
        functions: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> str:
        disable_stream = cfg.get("DISABLE_STREAMING") == "true"
        messages = self.make_messages(prompt.strip())

        generator = self.get_completion(
            model=model,
            temperature=temperature,
            top_p=top_p,
            messages=messages,
            functions=functions,
            **kwargs,
        )
        result = self.printer(generator, not disable_stream)
        # 即使 get_completion 内部已保存，此处再保存一次确保万无一失
        self._save_session()
        return result

    @classmethod
    def clear_session(cls, session_name: str = None):
        name = session_name or cls._current_session
        if name in cls._conversations:
            del cls._conversations[name]
        safe_name = Path(name).name
        file_path = cls.SESSIONS_DIR / f"{safe_name}.json"
        if file_path.exists():
            file_path.unlink()
        if name == cls._current_session:
            cls._current_session = "default"