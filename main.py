import argparse
import sys

from retriever import fetch_papers
from reranker import load_models, ensemble_rerank, ENSEMBLE_MODELS
from evaluate import evaluate, print_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="paperrank",
        description="Rank academic papers with OpenAlex + CrossEncoder ensemble reranking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument(
        "--limit", type=int, default=50,
        help="Number of candidate papers to fetch from OpenAlex",
    )
    parser.add_argument(
        "--k", type=int, default=10,
        help="Top-k cutoff for display and evaluation metrics (NDCG@k, MRR)",
    )
    parser.add_argument(
        "--relevant-ids", nargs="*", default=None, metavar="PAPER_ID",
        help="OpenAlex paper IDs to treat as relevant (for NDCG/MRR). "
             "If omitted, pseudo-relevance (OpenAlex top-k) is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Stage 1: Fetch ───────────────────────────────────────────────────────
    print(f"\n[1/3] Fetching papers from OpenAlex ...")
    try:
        baseline_papers = fetch_papers(args.query, limit=args.limit)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"      Retrieved {len(baseline_papers)} papers with abstracts.")

    # ── Stage 2: Rerank ──────────────────────────────────────────────────────
    print(f"\n[2/3] Running ensemble reranking ({len(ENSEMBLE_MODELS)} models) ...")
    models = load_models()
    print(f"      Scoring {len(baseline_papers)} papers × {len(ENSEMBLE_MODELS)} models ...")
    reranked_papers = ensemble_rerank(args.query, baseline_papers, models)

    # ── Stage 3: Evaluate & Display ──────────────────────────────────────────
    print(f"\n[3/3] Evaluating results ...\n")
    relevant_ids = set(args.relevant_ids) if args.relevant_ids else None
    metrics = evaluate(baseline_papers, reranked_papers, relevant_ids=relevant_ids, k=args.k)
    print_results(args.query, baseline_papers, reranked_papers, metrics, k=args.k, ensemble=True)


if __name__ == "__main__":
    main()
