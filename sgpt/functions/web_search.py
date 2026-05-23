import requests
from pydantic import BaseModel, Field
from typing import Dict, Any

class Function(BaseModel):
    """
    搜索互联网，返回相关结果的标题、摘要和链接。
    当用户询问最新信息、实时数据或需要查找未知内容时调用此函数。
    """

    query: str = Field(..., description="要搜索的关键词或问题，例如 'Python 3.13 发布时间'。")

    @classmethod
    def execute(cls, query: str) -> str:
        # 这里用 DuckDuckGo 的 API 作为示例（如果你能访问的话）
        # 若不能访问，请替换为你实际使用的搜索实现（如博查 API 或 scrape-bing 库）
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            abstract = data.get("AbstractText", "")
            related = [topic.get("Text", "") for topic in data.get("RelatedTopics", [])[:3]]
            results = []
            if abstract:
                results.append(f"摘要：{abstract}")
            if related:
                results.append("相关条目：\n" + "\n".join(f"- {r}" for r in related))
            return "\n".join(results) if results else "未找到搜索结果。"
        except Exception as e:
            return f"搜索出错：{e}"

    @classmethod
    def openai_schema(cls) -> Dict[str, Any]:
        schema = cls.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": cls.__doc__.strip() if cls.__doc__ else "",
                "parameters": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                },
            },
        }