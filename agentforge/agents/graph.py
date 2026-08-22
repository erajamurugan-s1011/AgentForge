import uuid
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from agentforge.agents.guardrail import run_guardrail
from agentforge.agents.planner import run_planner
from agentforge.agents.retriever import run_retriever
from agentforge.agents.executor import run_executor
from agentforge.agents.critique import run_critique
from fastmcp import Client
from agentforge.mcp_server.server import mcp

CONFIDENCE_THRESHOLD = 0.6


class AgentState(TypedDict):
    ticket_text: str
    trace_id: str
    is_safe: Optional[bool]
    unsafe_reason: Optional[str]
    category: Optional[str]
    priority: Optional[str]
    status_check_system: Optional[str]
    retrieved_docs: list
    draft_response: Optional[str]
    tool_results: dict
    confidence_score: Optional[float]
    unsupported_claims: list
    final_action: Optional[str]
    escalation_ticket_id: Optional[str]


def guardrail_node(state: AgentState) -> dict:
    verdict = run_guardrail(state["ticket_text"])
    return {"is_safe": verdict["is_safe"], "unsafe_reason": verdict.get("reason", "")}


def blocked_node(state: AgentState) -> dict:
    return {
        "final_action": "blocked",
        "draft_response": "This request could not be processed for security reasons. Please contact IT directly if you have a genuine support need.",
        "confidence_score": 0.0,
    }


def route_after_guardrail(state: AgentState) -> str:
    return "planner" if state["is_safe"] else "blocked"


def planner_node(state: AgentState) -> dict:
    plan = run_planner(state["ticket_text"])
    return {
        "trace_id": str(uuid.uuid4()),
        "category": plan["category"],
        "priority": plan["priority"],
        "status_check_system": plan.get("status_check_system"),
    }


async def retriever_node(state: AgentState) -> dict:
    docs = await run_retriever(state["ticket_text"], state["category"])
    return {"retrieved_docs": docs}


async def executor_node(state: AgentState) -> dict:
    result = await run_executor(
        state["ticket_text"], state["retrieved_docs"], state.get("status_check_system")
    )
    return {"draft_response": result["draft_response"], "tool_results": result["tool_results"]}


def critique_node(state: AgentState) -> dict:
    kb_context = "\n\n".join(
        f"[{d['kb_id']}] {d['title']}\n{d['content']}" for d in state["retrieved_docs"]
    )
    status_context = ""
    if state["tool_results"].get("status_check"):
        s = state["tool_results"]["status_check"]
        status_context = f"\n\nLive status: {s['status']}. {s.get('note', '')}"

    verdict = run_critique(state["draft_response"], kb_context, status_context)
    return {
        "confidence_score": verdict["confidence_score"],
        "unsupported_claims": verdict["unsupported_claims"],
    }


def resolve_node(state: AgentState) -> dict:
    return {"final_action": "resolved"}


async def escalate_node(state: AgentState) -> dict:
    summary = f"{state['ticket_text']}\n\nDraft attempted: {state['draft_response']}\nUnsupported claims: {state['unsupported_claims']}"
    async with Client(mcp) as client:
        result = await client.call_tool(
            "create_escalation",
            {"summary": summary, "priority": state["priority"], "category": state["category"]},
        )
    return {"final_action": "escalated", "escalation_ticket_id": result.data["ticket_id"]}


def route_after_critique(state: AgentState) -> str:
    return "resolve" if state["confidence_score"] >= CONFIDENCE_THRESHOLD else "escalate"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("blocked", blocked_node)
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("executor", executor_node)
    graph.add_node("critique", critique_node)
    graph.add_node("resolve", resolve_node)
    graph.add_node("escalate", escalate_node)

    graph.set_entry_point("guardrail")
    graph.add_conditional_edges("guardrail", route_after_guardrail, {"planner": "planner", "blocked": "blocked"})
    graph.add_edge("blocked", END)
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "executor")
    graph.add_edge("executor", "critique")
    graph.add_conditional_edges("critique", route_after_critique, {"resolve": "resolve", "escalate": "escalate"})
    graph.add_edge("resolve", END)
    graph.add_edge("escalate", END)

    return graph.compile()


agentforge_graph = build_graph()