from fastapi import FastAPI
from pydantic import BaseModel
from agentforge.agents.graph import agentforge_graph
from agentforge.mcp_server.server import get_embed_model, get_qdrant_client

app = FastAPI(title="AgentForge - IT Helpdesk Copilot", version="1.0.0")


class TicketRequest(BaseModel):
    ticket_text: str


class TicketResponse(BaseModel):
    trace_id: str | None = None
    is_safe: bool
    unsafe_reason: str | None = None
    category: str | None = None
    priority: str | None = None
    draft_response: str
    confidence_score: float
    final_action: str
    escalation_ticket_id: str | None = None


@app.on_event("startup")
async def warm_up():
    print("Warming up: loading embedding model...")
    try:
        get_embed_model()
        print("Embedding model loaded.")
    except Exception as e:
        print(f"Warning: embedding model failed to preload: {e}")

    try:
        get_qdrant_client().get_collections()
        print("Qdrant connection verified.")
    except Exception as e:
        print(f"Warning: Qdrant warm-up check failed (will retry on first real request): {e}")

    print("Warm-up complete.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ticket", response_model=TicketResponse)
async def submit_ticket(request: TicketRequest):
    result = await agentforge_graph.ainvoke({"ticket_text": request.ticket_text})
    return TicketResponse(
        trace_id=result.get("trace_id"),
        is_safe=result["is_safe"],
        unsafe_reason=result.get("unsafe_reason"),
        category=result.get("category"),
        priority=result.get("priority"),
        draft_response=result["draft_response"],
        confidence_score=result["confidence_score"],
        final_action=result["final_action"],
        escalation_ticket_id=result.get("escalation_ticket_id"),
    )