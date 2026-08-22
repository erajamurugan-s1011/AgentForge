import os
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
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        _qdrant_client = QdrantClient(host=qdrant_host, port=6333)
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


import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

ESCALATION_QUEUE_FILE = Path(__file__).parent.parent / "data" / "escalation_queue.json"

_SYSTEM_STATUS = {
    "vpn_gateway": {"status": "operational", "note": ""},
    "wifi_network": {"status": "operational", "note": ""},
    "email_server": {"status": "degraded", "note": "Some users seeing delayed delivery, ETA 2h"},
    "confluence_wiki": {"status": "operational", "note": ""},
    "jira": {"status": "operational", "note": ""},
    "license_server": {"status": "operational", "note": ""},
}


@mcp.tool
def check_status(system: str) -> dict:
    """Check the current operational status of an internal IT system.

    Args:
        system: One of vpn_gateway, wifi_network, email_server, confluence_wiki, jira, license_server.
    """
    return _SYSTEM_STATUS.get(
        system, {"status": "unknown", "note": f"'{system}' is not a recognized system name"}
    )


@mcp.tool
def create_escalation(summary: str, priority: str, category: str) -> dict:
    """Escalate a ticket to the human L2 support queue.

    Args:
        summary: A concise summary of the issue and what was already tried.
        priority: One of low, medium, high.
        category: One of network, access, hardware, software, other.
    """
    ESCALATION_QUEUE_FILE.parent.mkdir(exist_ok=True)
    queue = []
    if ESCALATION_QUEUE_FILE.exists():
        queue = json.loads(ESCALATION_QUEUE_FILE.read_text())

    ticket_id = f"ESC-{uuid.uuid4().hex[:8]}"
    entry = {
        "ticket_id": ticket_id,
        "summary": summary,
        "priority": priority,
        "category": category,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending_l2_review",
    }
    queue.append(entry)
    ESCALATION_QUEUE_FILE.write_text(json.dumps(queue, indent=2))

    return {"ticket_id": ticket_id, "status": "escalated"}


if __name__ == "__main__":
    mcp.run()