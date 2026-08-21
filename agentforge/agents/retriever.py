from fastmcp import Client
from agentforge.mcp_server.server import mcp


async def run_retriever(ticket_text: str, category: str = "", top_k: int = 3) -> list[dict]:
    async with Client(mcp) as client:
        result = await client.call_tool(
            "kb_search",
            {"query": ticket_text, "category": category, "top_k": top_k},
        )
        return result.data