from pydantic import BaseModel, Field
from typing import Optional, Literal


class TicketInput(BaseModel):
    ticket_text: str
    employee_id: Optional[str] = None


class TicketClassification(BaseModel):
    category: Literal["network", "access", "hardware", "software", "other"]
    priority: Literal["low", "medium", "high"]
    reasoning: str


class KBChunk(BaseModel):
    content: str
    source: str
    score: float


class AgentState(BaseModel):
    ticket_text: str
    category: Optional[str] = None
    priority: Optional[str] = None
    tool_plan: list[str] = Field(default_factory=list)
    retrieved_docs: list[KBChunk] = Field(default_factory=list)
    tool_results: dict = Field(default_factory=dict)
    draft_response: Optional[str] = None
    confidence_score: Optional[float] = None
    unsupported_claims: list[str] = Field(default_factory=list)
    final_action: Optional[Literal["resolved", "escalated"]] = None
    trace_id: Optional[str] = None