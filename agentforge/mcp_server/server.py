from fastmcp import FastMCP
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

mcp = FastMCP("agentforge-helpdesk-tools")

_embed_model = None
_qdrant_client = None
COLLECTION_NAME = "helpdesk_kb"


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(host="localhost", port=6333)
    return _qdrant_client


@mcp.tool
def kb_search(query: str, category: str = "", top_k: int = 3) -> list[dict]:
    """Search the IT helpdesk knowledge base for articles relevant to a query.

    Args:
        query: The search query, typically the employee's issue description.
        category: Optional filter - one of network, access, hardware, software, other.
        top_k: Number of results to return.
    """
    model = get_embed_model()
    client = get_qdrant_client()
    vector = model.encode(query).tolist()

    search_filter = None
    if category:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        search_filter = Filter(
            must=[FieldCondition(key="category", match=MatchValue(value=category))]
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        query_filter=search_filter,
        limit=top_k,
    ).points

    return [
        {
            "kb_id": r.payload["kb_id"],
            "title": r.payload["title"],
            "content": r.payload["content"],
            "score": round(r.score, 4),
        }
        for r in results
    ]


if __name__ == "__main__":
    mcp.run()