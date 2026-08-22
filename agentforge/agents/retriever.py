from fastmcp import Client
from agentforge.mcp_server.server import mcp

WEAK_SCORE_THRESHOLD = 0.3


async def run_retriever(ticket_text: str, category: str = "", top_k: int = 3) -> list[dict]:
    async with Client(mcp) as client:
        docs = []
        if category:
            result = await client.call_tool(
                "kb_search",
                {"query": ticket_text, "category": category, "top_k": top_k},
            )
            docs = result.data

        top_score = max((d["score"] for d in docs), default=0.0)
        if not docs or top_score < WEAK_SCORE_THRESHOLD:
            # Category filter produced nothing useful (likely a borderline
            # misclassification) - fall back to an unfiltered search rather
            # than returning no context at all.
            fallback = await client.call_tool(
                "kb_search",
                {"query": ticket_text, "category": "", "top_k": top_k},
            )
            fallback_docs = fallback.data
            fallback_top = max((d["score"] for d in fallback_docs), default=0.0)
            if fallback_top > top_score:
                docs = fallback_docs

        return docs