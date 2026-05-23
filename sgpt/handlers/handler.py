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

# ---------- API 调用方式 ----------
completion: Callable[..., Any] = lambda *args, **kwargs: Generator[Any, None, None]

base_url = cfg.get("API_BASE_URL")
use_litellm = cfg.get("USE_LITELLM") == "true"

base_request_kwargs = {
    "timeout": int(cfg.get("REQUEST_TIMEOUT")),
    "api_key": cfg.get("OPENAI_API_KEY"),
    "base_url": None if base_url == "default" else base_url,
}

if use_litellm:
    import litellm

    completion = litellm.completion
    litellm.suppress_debug_info = True
else:
    from openai import OpenAI

    client = OpenAI(**base_request_kwargs)
    completion = client.chat.completions.create
    base_request_kwargs.clear()


class Handler:
    cache = Cache(int(cfg.get("CACHE_LENGTH")), Path(cfg.get("CACHE_PATH")))

    def __init__(self, role: SystemRole, markdown: bool) -> None:
        self.role = role
        api_base_url = cfg.get("API_BASE_URL")
        self.base_url = None if api_base_url == "default" else api_base_url
        self.timeout = int(cfg.get("REQUEST_TIMEOUT"))
        self.markdown = "APPLY MARKDOWN" in self.role.role and markdown
        self.code_theme = cfg.get("CODE_THEME")
        self.color = cfg.get("DEFAULT_COLOR")

    @property
    def printer(self) -> Printer:
        if self.markdown:
            return MarkdownPrinter(self.code_theme)
        return TextPrinter(self.color)

    def make_messages(self, prompt: str) -> List[Dict[str, str]]:
        raise NotImplementedError

    def _execute_tool_call(
        self,
        messages: List[Dict[str, Any]],
        tool_call: dict,
    ) -> Generator[str, None, None]:
        """
        执行单个工具调用（包含确认逻辑），并添加 tool 消息。
        返回生成器，输出执行过程中的提示信息。
        """
        tool_call_id = tool_call["id"]
        name = tool_call["function"]["name"]
        arguments = tool_call["function"]["arguments"]

        dict_args = json.loads(arguments)
        joined_args = ", ".join(f'{k}="{v}"' for k, v in dict_args.items())
        yield f"> @FunctionCall `{name}({joined_args})` \n\n"

        # 用户确认
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
                yield "**User denied the function call.**\n"

        if execute_allowed:
            result = get_function(name)(**dict_args)
            if cfg.get("SHOW_FUNCTIONS_OUTPUT") == "true":
                yield f"```text\n{result}\n```\n"
        else:
            result = "User denied the function call."

        messages.append(
            {"role": "tool", "content": result, "tool_call_id": tool_call_id}
        )

    @cache
    def get_completion(
        self,
        model: str,
        temperature: float,
        top_p: float,
        messages: List[Dict[str, Any]],
        functions: Optional[List[Dict[str, str]]],
        caching: bool = True,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        reasoning_content = ""
        tool_calls: List[dict] = []

        # 角色限制
        is_shell_role = self.role.name == DefaultRoles.SHELL.value
        is_code_role = self.role.name == DefaultRoles.CODE.value
        is_dsc_shell_role = self.role.name == DefaultRoles.DESCRIBE_SHELL.value
        if is_shell_role or is_code_role or is_dsc_shell_role:
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

                # 思维链内容
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    reasoning_content += delta.reasoning_content
                elif isinstance(delta, dict) and delta.get("reasoning_content"):
                    reasoning_content += delta["reasoning_content"]

                # 工具调用 delta
                delta_tool_calls = (
                    delta.get("tool_calls") if use_litellm else delta.tool_calls
                )
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

                # 正常内容输出
                if delta.content:
                    yield delta.content

                # 流结束且包含工具调用
                if chunk.choices[0].finish_reason == "tool_calls":
                    assistant_msg = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": tc["function"],
                            }
                            for tc in tool_calls
                        ],
                    }
                    if reasoning_content:
                        assistant_msg["reasoning_content"] = reasoning_content
                    messages.append(assistant_msg)

                    yield "\n"
                    for tc in tool_calls:
                        yield from self._execute_tool_call(messages, tc)

                    yield from self.get_completion(
                        model=model,
                        temperature=temperature,
                        top_p=top_p,
                        messages=messages,
                        functions=functions,
                        caching=False,
                    )
                    return

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
            caching=caching,
            **kwargs,
        )
        return self.printer(generator, not disable_stream)