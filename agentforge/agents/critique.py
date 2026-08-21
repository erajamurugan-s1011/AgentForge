import json
from agentforge.utils.llm import get_llm

CRITIQUE_SYSTEM_PROMPT = """You are a strict fact-checking agent reviewing a draft IT helpdesk response.
Your job is to verify every factual claim in the draft is actually supported by the provided context
(knowledge base articles and system status). Be skeptical - if a step, number, or system name in the
draft isn't clearly backed by the context, flag it.

Respond ONLY with valid JSON in this exact shape, no other text:
{
  "confidence_score": <float 0.0 to 1.0>,
  "unsupported_claims": ["claim 1 not backed by context", ...],
  "verdict_reasoning": "one sentence explaining the score"
}

Scoring guide:
- 1.0 = every claim is directly traceable to the provided context
- 0.5-0.8 = mostly grounded but some minor unsupported additions (tone, filler) - still usually fine
- below 0.5 = contains steps or facts not present anywhere in the context - do not auto-send this
"""


def run_critique(draft_response: str, kb_context: str, status_context: str = "") -> dict:
    llm = get_llm(temperature=0)
    full_context = f"{kb_context}{status_context}"

    response = llm.invoke([
        {"role": "system", "content": CRITIQUE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Context provided to the drafting agent:\n{full_context}\n\nDraft response to check:\n{draft_response}"},
    ])

    content = response.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    content = content.strip()

    return json.loads(content)