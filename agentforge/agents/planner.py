import json
from agentforge.utils.llm import get_llm

PLANNER_SYSTEM_PROMPT = """You are the planning agent for an IT helpdesk assistant.
Given an employee's ticket, classify it and decide what to do next.

Respond ONLY with valid JSON in this exact shape, no other text:
{
  "category": "network" | "access" | "hardware" | "software" | "other",
  "priority": "low" | "medium" | "high",
  "reasoning": "one sentence explaining the classification",
  "needs_status_check": true | false,
  "status_check_system": "vpn_gateway" | "wifi_network" | "email_server" | "confluence_wiki" | "jira" | "license_server" | null
}

Rules:
- priority "high" only for things blocking work entirely (cannot log in, laptop dead, no network at all)
- "needs_status_check" is true only if checking a system's live status would materially help (e.g. VPN issues, email not syncing)
- If needs_status_check is true, status_check_system must be one of the listed systems that matches the issue
"""


def run_planner(ticket_text: str) -> dict:
    llm = get_llm(temperature=0)
    response = llm.invoke([
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": ticket_text},
    ])

    content = response.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    content = content.strip()

    plan = json.loads(content)
    return plan