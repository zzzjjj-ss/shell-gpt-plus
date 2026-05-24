import json
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, Field

MEMORY_FILE = Path.home() / ".shell_gpt_memory.txt"


class Function(BaseModel):
    info: str = Field(
        ...,
        description="需要记住的信息，例如 '记住我的工具目录是 E:\桌面\文件\ffmpeg-2023-08-17-git-9ae4863cc5-full_build\bin'",
    )

    @classmethod
    def execute(cls, info: str) -> str:
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(info.strip() + "\n")
        return f"已记住：{info}"

    @classmethod
    def openai_schema(cls) -> Dict[str, Any]:
        schema = cls.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": "remember",
                "description": cls.__doc__.strip() if cls.__doc__ else "",
                "parameters": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                },
            },
        }