"""PaperRank command line: the same pipeline as the web app, printed to the terminal.

    python main.py "effects of microplastics on gut microbiome in zebrafish larvae"
"""
import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from analyzer import LLMError, assess_coverage, decompose, find_gaps
from reranker import AVAILABLE_MODELS, DEFAULT_MODEL, load_model, rerank
from retriever import multi_search


STATUS_MARK = {"covered": "✓", "partial": "~", "missing": "✗"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="paperrank",
        description="Find papers for a research question, check aspect coverage, and spot possible gaps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("question", type=str, help="Research question in plain language")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=AVAILABLE_MODELS, help="Reranker model")
    parser.add_argument("--top", type=int, default=5, help="Papers to analyze and display")
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    args = parse_args()

    print("[1/4] Breaking the question into aspects ...")
    plan = decompose(args.question)
    print("      Aspects    :", ", ".join(a["name"] for a in plan["aspects"]) or "(none: LLM unavailable)")
    print("      Sub-queries:", *plan["sub_queries"], sep="\n        - ")

    print("[2/4] Searching OpenAlex ...")
    papers = multi_search(plan["sub_queries"])
    if not papers:
        print("No papers with abstracts found. Try rephrasing.", file=sys.stderr)
        sys.exit(1)

    print(f"[3/4] Reranking {len(papers)} candidates with {args.model} ...")
    top = rerank(args.question, papers, load_model(args.model))[:args.top]

    print("[4/4] Checking coverage ...")
    try:
        assessments = assess_coverage(args.question, plan["aspects"], top)
    except LLMError as e:
        print(f"      Coverage unavailable: {e}")
        assessments = [None] * len(top)

    for paper, r in zip(top, assessments):
        coverage = f"{r['coverage']}% coverage" if r and r["coverage"] is not None else "not analyzed"
        print(f"\n#{paper['ce_rank']} [{coverage}, match {paper['ce_score']:.2f}] {paper['title']}")
        print(f"   {paper.get('year') or 'n.d.'} · {paper.get('citationCount') or 0} citations · {paper.get('url', '')}")
        for a in (r or {}).get("aspects", []):
            print(f"   {STATUS_MARK[a['status']]} {a['name']}" + (f'  "{a["evidence"]}"' if a["evidence"] else ""))
        if r and r["note"]:
            print(f"   not addressed: {r['note']}")

    if all(assessments):
        gaps = find_gaps(plan["aspects"], assessments)
        print("\nPossible gaps (leads to verify, not proof)")
        print("  Not fully covered by any top paper:", ", ".join(gaps["uncovered"]) or "none")
        if gaps["full_count"] is not None:
            print(f"  All aspects together ({gaps['full_query']!r}): {gaps['full_count']} papers")
            for b in gaps["bottlenecks"]:
                ratio = b["count_without"] / max(gaps["full_count"], 1)
                print(f"    without '{b['aspect']}': {b['count_without']} papers ({ratio:.0f}x more)")


if __name__ == "__main__":
    main()
