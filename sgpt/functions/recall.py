import json
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, Field

MEMORY_FILE = Path.home() / ".shell_gpt_memory.txt"


class Function(BaseModel):
    query: str = Field(
        ...,
        description="用于回忆记忆的关键词或问题（所有记忆都会被返回）",
    )

    @classmethod
    def execute(cls, query: str) -> str:
        if not MEMORY_FILE.exists():
            return "当前没有任何保存的记忆。"
        content = MEMORY_FILE.read_text(encoding="utf-8")
        if not content.strip():
            return "当前没有任何保存的记忆。"
        return f"已保存的记忆：\n{content.strip()}"

    @classmethod
    def openai_schema(cls) -> Dict[str, Any]:
        schema = cls.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": "recall",
                "description": cls.__doc__.strip() if cls.__doc__ else "",
                "parameters": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                },
            },
        }