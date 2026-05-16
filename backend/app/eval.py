"""
Evaluation script: replays sample conversations against the /chat endpoint.

Tracks:
  - Schema validity
  - Catalog-only URLs
  - Recommendation count (0 or 1-10)
  - Clarification behavior
  - Refinement behavior
  - Comparison behavior
  - Refusal behavior
  - Latency per turn
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:8000"
CONVERSATIONS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sample_conversations" / "GenAI_SampleConversations"

# Load catalog URLs for validation
CATALOG_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "shl_catalog.json"


def load_catalog_urls() -> set[str]:
    """Load all valid catalog URLs."""
    with open(CATALOG_PATH, encoding="utf-8") as f:
        items = json.load(f)
    return {item["link"] for item in items if item.get("status") == "ok"}


def parse_conversation(filepath: Path) -> list[dict]:
    """
    Parse a sample conversation markdown file into turns.

    Returns list of dicts with:
      - turn_number: int
      - user_message: str
      - expected_has_recs: bool (whether the agent showed a table)
      - expected_end: bool
      - expected_assessment_names: list[str] (from table rows)
    """
    text = filepath.read_text(encoding="utf-8")
    turns = []

    # Split by ### Turn N
    turn_blocks = re.split(r"### Turn \d+", text)[1:]

    for i, block in enumerate(turn_blocks):
        # Extract user message
        user_match = re.search(r"\*\*User\*\*\s*\n\s*>(.*?)(?=\n\*\*Agent\*\*|\Z)", block, re.DOTALL)
        if not user_match:
            continue

        user_msg = user_match.group(1).strip()
        # Clean up multi-line blockquotes
        user_msg = re.sub(r"\n\s*>\s*", "\n", user_msg).strip()

        # Check if agent response has a table (recommendations)
        has_table = bool(re.search(r"\|.*\|.*\|.*\|", block))

        # Check end_of_conversation
        end_match = re.search(r"end_of_conversation.*\*\*(\w+)\*\*", block)
        expected_end = end_match.group(1).lower() == "true" if end_match else False

        # Extract expected assessment names from table rows
        expected_names = []
        table_rows = re.findall(r"\|\s*\d+\s*\|\s*([^|]+)\s*\|", block)
        for name in table_rows:
            expected_names.append(name.strip())

        turns.append({
            "turn_number": i + 1,
            "user_message": user_msg,
            "expected_has_recs": has_table,
            "expected_end": expected_end,
            "expected_assessment_names": expected_names,
        })

    return turns


def replay_conversation(
    filepath: Path,
    catalog_urls: set[str],
    client: httpx.Client,
) -> dict:
    """Replay a single conversation and collect metrics."""
    conv_name = filepath.stem
    turns = parse_conversation(filepath)
    messages: list[dict] = []
    results = {
        "conversation": conv_name,
        "total_turns": len(turns),
        "turn_results": [],
        "all_schema_valid": True,
        "all_urls_valid": True,
        "total_latency_s": 0.0,
    }

    for turn in turns:
        # Add user message
        messages.append({"role": "user", "content": turn["user_message"]})

        # Call /chat
        start = time.time()
        try:
            resp = client.post(
                f"{BASE_URL}/chat",
                json={"messages": messages},
                timeout=60.0,
            )
            latency = time.time() - start
        except Exception as e:
            results["turn_results"].append({
                "turn": turn["turn_number"],
                "error": str(e),
                "latency_s": time.time() - start,
            })
            results["all_schema_valid"] = False
            continue

        results["total_latency_s"] += latency

        # Parse response
        turn_result: dict = {
            "turn": turn["turn_number"],
            "user_message": turn["user_message"][:60],
            "latency_s": round(latency, 2),
        }

        if resp.status_code != 200:
            turn_result["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
            turn_result["schema_valid"] = False
            results["all_schema_valid"] = False
            results["turn_results"].append(turn_result)
            continue

        data = resp.json()

        # Schema validation
        schema_valid = (
            isinstance(data.get("reply"), str)
            and isinstance(data.get("recommendations"), list)
            and isinstance(data.get("end_of_conversation"), bool)
        )
        turn_result["schema_valid"] = schema_valid
        if not schema_valid:
            results["all_schema_valid"] = False

        # Recommendations checks
        recs = data.get("recommendations", [])
        turn_result["rec_count"] = len(recs)
        turn_result["expected_has_recs"] = turn["expected_has_recs"]
        turn_result["actual_has_recs"] = len(recs) > 0
        turn_result["rec_count_valid"] = len(recs) == 0 or 1 <= len(recs) <= 10

        # URL validation
        invalid_urls = []
        for rec in recs:
            url = rec.get("url", "")
            if url and url not in catalog_urls:
                invalid_urls.append(url)
        turn_result["invalid_urls"] = invalid_urls
        if invalid_urls:
            results["all_urls_valid"] = False

        # Expected assessment coverage
        if turn["expected_assessment_names"]:
            found_names = {r.get("name", "").lower() for r in recs}
            expected_found = 0
            for exp_name in turn["expected_assessment_names"]:
                if any(exp_name.lower() in fn or fn in exp_name.lower() for fn in found_names):
                    expected_found += 1
            turn_result["expected_names"] = len(turn["expected_assessment_names"])
            turn_result["found_expected"] = expected_found
            turn_result["coverage"] = (
                round(expected_found / len(turn["expected_assessment_names"]) * 100, 1)
                if turn["expected_assessment_names"]
                else 0
            )

        # End of conversation
        turn_result["expected_end"] = turn["expected_end"]
        turn_result["actual_end"] = data.get("end_of_conversation", False)

        # Latency check
        turn_result["under_30s"] = latency < 30.0

        # Reply snippet
        turn_result["reply_snippet"] = data.get("reply", "")[:100]

        results["turn_results"].append(turn_result)

        # Add assistant response to history
        messages.append({"role": "assistant", "content": data.get("reply", "")})

        # Avoid 429 Rate Limits
        time.sleep(1.0)

    return results


def main():
    """Run evaluation across all sample conversations."""
    catalog_urls = load_catalog_urls()
    print(f"Loaded {len(catalog_urls)} valid catalog URLs\n")

    conv_files = sorted(CONVERSATIONS_DIR.glob("C*.md"))
    if not conv_files:
        print(f"No conversation files found in {CONVERSATIONS_DIR}")
        sys.exit(1)

    print(f"Found {len(conv_files)} conversations to evaluate\n")
    print("=" * 80)

    all_results = []
    overall_stats = {
        "total_conversations": 0,
        "total_turns": 0,
        "schema_valid_turns": 0,
        "url_valid_turns": 0,
        "rec_count_valid_turns": 0,
        "under_30s_turns": 0,
        "correct_rec_presence": 0,
        "total_latency": 0.0,
    }

    with httpx.Client() as client:
        # Health check
        try:
            health = client.get(f"{BASE_URL}/health", timeout=5.0)
            print(f"Health check: {health.json()}\n")
        except Exception as e:
            print(f"ERROR: Server not reachable at {BASE_URL}: {e}")
            sys.exit(1)

        for conv_file in conv_files:
            print(f"\n--- {conv_file.stem} ---")
            result = replay_conversation(conv_file, catalog_urls, client)
            all_results.append(result)

            overall_stats["total_conversations"] += 1
            overall_stats["total_latency"] += result["total_latency_s"]

            for tr in result["turn_results"]:
                overall_stats["total_turns"] += 1

                if tr.get("schema_valid", False):
                    overall_stats["schema_valid_turns"] += 1
                if not tr.get("invalid_urls"):
                    overall_stats["url_valid_turns"] += 1
                if tr.get("rec_count_valid", False):
                    overall_stats["rec_count_valid_turns"] += 1
                if tr.get("under_30s", False):
                    overall_stats["under_30s_turns"] += 1
                if tr.get("expected_has_recs") == tr.get("actual_has_recs"):
                    overall_stats["correct_rec_presence"] += 1

                # Print turn summary
                status = "[OK]" if tr.get("schema_valid") else "[FAIL]"
                rec_info = f"recs={tr.get('rec_count', 'N/A')}"
                latency = f"{tr.get('latency_s', 0):.1f}s"
                coverage = f"coverage={tr.get('coverage', 'N/A')}%" if "coverage" in tr else ""
                print(f"  Turn {tr.get('turn')}: {status} | {latency} | {rec_info} | {coverage}")
                if tr.get("invalid_urls"):
                    print(f"    ⚠ Invalid URLs: {tr['invalid_urls']}")
                if tr.get("error"):
                    print(f"    ✗ Error: {tr['error']}")

    # Final summary
    total = overall_stats["total_turns"]
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Conversations:       {overall_stats['total_conversations']}")
    print(f"Total turns:         {total}")
    print(f"Schema valid:        {overall_stats['schema_valid_turns']}/{total}")
    print(f"URL valid:           {overall_stats['url_valid_turns']}/{total}")
    print(f"Rec count valid:     {overall_stats['rec_count_valid_turns']}/{total}")
    print(f"Under 30s:           {overall_stats['under_30s_turns']}/{total}")
    print(f"Correct rec present: {overall_stats['correct_rec_presence']}/{total}")
    print(f"Total latency:       {overall_stats['total_latency']:.1f}s")
    print(f"Avg latency/turn:    {overall_stats['total_latency'] / max(total, 1):.1f}s")

    # Save detailed results
    output_path = Path(__file__).resolve().parent.parent.parent / "data" / "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
