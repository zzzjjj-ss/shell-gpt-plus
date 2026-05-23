import subprocess
import locale
import re
from typing import Any, ClassVar, Dict, List  # 新增 ClassVar, List 导入

from pydantic import BaseModel, Field


class Function(BaseModel):
    """
    执行一个 Shell 命令并返回输出（结果）。
    注意：危险命令会被自动拦截，确保系统安全。
    """

    shell_command: str = Field(
        ...,
        example="dir",
        description="要执行的 shell 命令。",
    )

    # 关键修改：添加 ClassVar[List[str]] 类型注解
    DANGEROUS_PATTERNS: ClassVar[List[str]] = [
        r"\bdel\b",
        r"\berase\b",
        r"\brmdir\b",
        r"\brd\b",
        r"\bformat\b",
        r"\bshutdown\b",
        r"\brestart\b",
        r"\blogoff\b",
        r"\breg\s+delete\b",
        r"\breg\s+add\b.*\/f",
        r"\bicacls\b",
        r"\btakeown\b",
        r"\bnet\s+user\b.*\/delete",
        r"\bnet\s+localgroup\b.*\/delete",
        r"\bsc\s+stop\b",
        r"\bsc\s+delete\b",
        r"\bwmic\s+process\s+call\s+create\b",
        r"\bschtasks\b.*\/delete",
        r"\bpowershell\s+.*-EncodedCommand\b",
        r"\bcmd\s+/c\s+.*del\b",
        r"&&\s*del\b",
        r"\|\s*del\b",
    ]

    @classmethod
    def _is_dangerous(cls, command: str) -> bool:
        """检查命令是否命中危险模式"""
        command_lower = command.lower()
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, command_lower):
                return True
        return False

    @classmethod
    def execute(cls, shell_command: str) -> str:
        if cls._is_dangerous(shell_command):
            return (
                "Error: Potentially dangerous command detected and blocked.\n"
                "If you believe this is a safe operation, please rephrase it "
                "or execute it manually with proper authorization."
            )
        process = subprocess.Popen(
            shell_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        output, _ = process.communicate()
        exit_code = process.returncode
        encoding = locale.getpreferredencoding()
        return f"Exit code: {exit_code}, Output:\n{output.decode(encoding, errors='replace')}"

    @classmethod
    def openai_schema(cls) -> Dict[str, Any]:
        schema = cls.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": "execute_shell_command",
                "description": cls.__doc__.strip() if cls.__doc__ else "",
                "parameters": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                },
            },
        }