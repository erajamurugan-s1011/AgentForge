import asyncio
import json
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from groq import RateLimitError

from agentforge.agents.graph import agentforge_graph
from agentforge.eval.test_tickets import TEST_TICKETS

RESULTS_FILE = Path(__file__).parent / "eval_results.json"


async def run_single(ticket: dict, embed_model: SentenceTransformer) -> dict:
    start = time.time()

    max_retries = 5
    for attempt in range(max_retries):
        try:
            result = await agentforge_graph.ainvoke({"ticket_text": ticket["ticket_text"]})
            break
        except RateLimitError:
            wait = 15 * (attempt + 1)
            print(f"  Rate limited, waiting {wait}s...")
            await asyncio.sleep(wait)
    else:
        raise RuntimeError(f"Failed after {max_retries} retries due to rate limiting")

    latency = round(time.time() - start, 2)

    category_correct = result["category"] == ticket["expected_category"]
    action_correct = result["final_action"] == ticket["expected_action"]

    q_emb = embed_model.encode(ticket["ticket_text"])
    a_emb = embed_model.encode(result.get("draft_response") or "")
    answer_relevance = float(cos_sim(q_emb, a_emb)[0][0])

    return {
        "id": ticket["id"],
        "ticket_text": ticket["ticket_text"],
        "expected_category": ticket["expected_category"],
        "predicted_category": result["category"],
        "category_correct": category_correct,
        "expected_action": ticket["expected_action"],
        "predicted_action": result["final_action"],
        "action_correct": action_correct,
        "faithfulness_score": result["confidence_score"],
        "unsupported_claims_count": len(result["unsupported_claims"]),
        "answer_relevance": round(answer_relevance, 4),
        "latency_sec": latency,
    }


async def main():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    results = []

    for i, ticket in enumerate(TEST_TICKETS, 1):
        print(f"[{i}/{len(TEST_TICKETS)}] Running {ticket['id']}: {ticket['ticket_text'][:60]}...")
        await asyncio.sleep(3)
        r = await run_single(ticket, embed_model)
        results.append(r)
        print(f"  -> category={r['predicted_category']} ({'OK' if r['category_correct'] else 'MISS'}), "
              f"action={r['predicted_action']} ({'OK' if r['action_correct'] else 'MISS'}), "
              f"faithfulness={r['faithfulness_score']}")

    n = len(results)
    category_acc = sum(r["category_correct"] for r in results) / n
    action_acc = sum(r["action_correct"] for r in results) / n
    avg_faithfulness = sum(r["faithfulness_score"] for r in results) / n
    avg_relevance = sum(r["answer_relevance"] for r in results) / n
    avg_latency = sum(r["latency_sec"] for r in results) / n
    zero_hallucination_rate = sum(r["unsupported_claims_count"] == 0 for r in results) / n

    summary = {
        "num_tickets": n,
        "category_classification_accuracy": round(category_acc, 4),
        "routing_accuracy": round(action_acc, 4),
        "avg_faithfulness_score": round(avg_faithfulness, 4),
        "avg_answer_relevance": round(avg_relevance, 4),
        "zero_unsupported_claims_rate": round(zero_hallucination_rate, 4),
        "avg_latency_sec": round(avg_latency, 2),
    }
    
    print("\n" + "=" * 50)
    print("EVAL SUMMARY")
    print("=" * 50)
    for k, v in summary.items():
        print(f"{k}: {v}")

    RESULTS_FILE.write_text(json.dumps({"summary": summary, "per_ticket": results}, indent=2))
    print(f"\nFull results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    asyncio.run(main())