import argparse
import sys

from retriever import retrieve
from reranker import load_model, rerank
from evaluate import evaluate, print_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="paperrank",
        description="Rank academic papers with BM25 + CrossEncoder reranking (Semantic Scholar)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument(
        "--limit", type=int, default=50,
        help="Number of candidate papers to fetch from Semantic Scholar",
    )
    parser.add_argument(
        "--k", type=int, default=10,
        help="Top-k cutoff for display and evaluation metrics (NDCG@k, MRR)",
    )
    parser.add_argument(
        "--model", type=str, default="BAAI/bge-reranker-v2-m3",
        help="HuggingFace CrossEncoder model ID",
    )
    parser.add_argument(
        "--relevant-ids", nargs="*", default=None, metavar="PAPER_ID",
        help="Semantic Scholar paper IDs to treat as relevant (for NDCG/MRR). "
             "If omitted, pseudo-relevance (BM25 top-k) is used.",
    )
    parser.add_argument(
        "--no-rerank", action="store_true",
        help="Skip CrossEncoder reranking — show BM25 results only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Stage 1: Retrieve ────────────────────────────────────────────────────
    print(f"\n[1/3] Fetching up to {args.limit} papers from Semantic Scholar ...")
    try:
        bm25_papers = retrieve(args.query, limit=args.limit)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"      Retrieved {len(bm25_papers)} papers with abstracts.")

    if args.no_rerank:
        # Print BM25-only results and exit
        print(f"\nBM25 top-{args.k} results for: {args.query!r}\n")
        for paper in bm25_papers[:args.k]:
            authors = ", ".join(a["name"] for a in paper.get("authors", [])[:2])
            print(f"[#{paper['bm25_rank']}] score={paper['bm25_score']}")
            print(f"  {paper['title']}")
            print(f"  {paper.get('year', 'N/A')} | {paper.get('citationCount', 'N/A')} citations | {authors}")
            print()
        return

    # ── Stage 2: Rerank ──────────────────────────────────────────────────────
    print(f"\n[2/3] Loading CrossEncoder model: {args.model}")
    model = load_model(args.model)

    print(f"      Scoring {len(bm25_papers)} (query, paper) pairs ...")
    reranked_papers = rerank(args.query, bm25_papers, model)

    # ── Stage 3: Evaluate & Display ──────────────────────────────────────────
    print(f"\n[3/3] Evaluating and displaying results ...\n")
    relevant_ids = set(args.relevant_ids) if args.relevant_ids else None
    metrics = evaluate(bm25_papers, reranked_papers, relevant_ids=relevant_ids, k=args.k)
    print_results(args.query, bm25_papers, reranked_papers, metrics, k=args.k)


if __name__ == "__main__":
    main()
