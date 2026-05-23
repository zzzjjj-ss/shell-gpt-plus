import requests
from pydantic import BaseModel, Field
from typing import Dict, Any
from bs4 import BeautifulSoup

class Function(BaseModel):
    """
    抓取指定 URL 的网页文本内容（提取正文）。
    当需要查看某个链接的具体内容时调用此函数。
    """

    url: str = Field(..., description="要抓取的网页完整网址，例如 'https://example.com'。")

    @classmethod
    def execute(cls, url: str) -> str:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            # 移除脚本和样式
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            # 限制长度，避免超出 token 限制
            return text[:3000]
        except Exception as e:
            return f"抓取网页出错：{e}"

    @classmethod
    def openai_schema(cls) -> Dict[str, Any]:
        schema = cls.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": "fetch_url",
                "description": cls.__doc__.strip() if cls.__doc__ else "",
                "parameters": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                },
            },
        }