from fastapi import FastAPI
from pydantic import BaseModel
from agentforge.agents.graph import agentforge_graph

app = FastAPI(title="AgentForge - IT Helpdesk Copilot", version="1.0.0")


class TicketRequest(BaseModel):
    ticket_text: str


class TicketResponse(BaseModel):
    trace_id: str
    category: str
    priority: str
    draft_response: str
    confidence_score: float
    final_action: str
    escalation_ticket_id: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ticket", response_model=TicketResponse)
async def submit_ticket(request: TicketRequest):
    result = await agentforge_graph.ainvoke({"ticket_text": request.ticket_text})
    return TicketResponse(
        trace_id=result["trace_id"],
        category=result["category"],
        priority=result["priority"],
        draft_response=result["draft_response"],
        confidence_score=result["confidence_score"],
        final_action=result["final_action"],
        escalation_ticket_id=result.get("escalation_ticket_id"),
    )