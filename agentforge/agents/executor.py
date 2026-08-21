from fastmcp import Client
from agentforge.mcp_server.server import mcp
from agentforge.utils.llm import get_llm

EXECUTOR_SYSTEM_PROMPT = """You are the executor agent for an IT helpdesk assistant.
Draft a clear, concise resolution message to the employee based ONLY on the knowledge base
articles and system status provided below. Do not invent steps, systems, or facts that
are not present in the provided context. If the context doesn't fully cover the issue,
say so honestly and note that it will be escalated.

Keep the tone friendly and direct, like a helpful IT colleague. 3-6 sentences max."""


async def run_executor(
    ticket_text: str,
    retrieved_docs: list[dict],
    status_check_system: str | None = None,
) -> dict:
    tool_results = {}

    if status_check_system:
        async with Client(mcp) as client:
            result = await client.call_tool("check_status", {"system": status_check_system})
            tool_results["status_check"] = result.data

    kb_context = "\n\n".join(
        f"[{d['kb_id']}] {d['title']}\n{d['content']}" for d in retrieved_docs
    )
    status_context = ""
    if tool_results.get("status_check"):
        s = tool_results["status_check"]
        status_context = f"\n\nLive system status ({status_check_system}): {s['status']}. {s.get('note', '')}"

    llm = get_llm(temperature=0.2)
    response = llm.invoke([
        {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT},
        {"role": "user", "content": f"Employee ticket: {ticket_text}\n\nKnowledge base context:\n{kb_context}{status_context}"},
    ])

    return {
        "draft_response": response.content.strip(),
        "tool_results": tool_results,
    }