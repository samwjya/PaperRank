import math
import pandas as pd


def dcg_at_k(relevances: list[float], k: int) -> float:
    """Compute Discounted Cumulative Gain at rank k."""
    total = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        total += rel / math.log2(i + 1)
    return total


def ndcg_at_k(ranked_relevances: list[float], k: int) -> float:
    """Compute Normalized DCG at rank k.

    Normalizes DCG by the ideal ordering (perfect ranking) of the same relevance labels.
    Returns a value in [0, 1]. Returns 0.0 if there are no relevant documents.
    """
    ideal = sorted(ranked_relevances, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(ranked_relevances, k) / idcg


def mrr(ranked_relevances: list[float]) -> float:
    """Compute Mean Reciprocal Rank.

    Returns 1/rank of the first relevant document (relevance > 0).
    Returns 0.0 if no relevant document is found.
    """
    for rank, rel in enumerate(ranked_relevances, start=1):
        if rel > 0:
            return 1.0 / rank
    return 0.0


def assign_relevance(papers: list[dict], relevant_ids: set[str]) -> list[float]:
    """Map a ranked list of papers to binary relevance labels.

    A paper is relevant (1.0) if its paperId is in relevant_ids, else 0.0.
    Used when the caller supplies explicit relevance judgments.
    """
    return [1.0 if p.get("paperId") in relevant_ids else 0.0 for p in papers]


def auto_relevance(papers: list[dict], query: str, top_n: int = 10) -> set[str]:
    """Derive a pseudo-relevance set from the top-N BM25 results.

    In the absence of manual judgments, we treat the top-N BM25 papers as
    relevant. This is a recall-based approximation — useful for demo and
    relative comparison, not absolute metric benchmarking.
    """
    bm25_sorted = sorted(papers, key=lambda p: p.get("bm25_rank", 9999))
    return {p["paperId"] for p in bm25_sorted[:top_n] if p.get("paperId")}


def build_comparison_table(bm25_papers: list[dict], reranked_papers: list[dict], top_k: int = 10) -> pd.DataFrame:
    """Build a side-by-side DataFrame comparing BM25 and reranked top-K results.

    Each row is one position (1..top_k). Columns show the BM25 paper at that
    rank and the reranked paper at that rank, so rank shifts are immediately visible.
    """
    rows = []
    for i in range(top_k):
        bm25_p = bm25_papers[i] if i < len(bm25_papers) else {}
        ce_p = reranked_papers[i] if i < len(reranked_papers) else {}

        bm25_title = (bm25_p.get("title") or "")[:60]
        ce_title = (ce_p.get("title") or "")[:60]

        rows.append({
            "Rank": i + 1,
            "BM25 Title": bm25_title,
            "BM25 Score": bm25_p.get("bm25_score", ""),
            "CE Title": ce_title,
            "CE Score": ce_p.get("ce_score", ""),
        })
    return pd.DataFrame(rows).set_index("Rank")


def evaluate(bm25_papers: list[dict], reranked_papers: list[dict],
             relevant_ids: set[str] | None = None, k: int = 10) -> dict:
    """Compute NDCG@k and MRR for both BM25 and reranked lists.

    If relevant_ids is None, falls back to auto_relevance (pseudo-relevance
    from BM25 top-k). Returns a dict with all four metrics.
    """
    if relevant_ids is None:
        relevant_ids = auto_relevance(bm25_papers, query="", top_n=k)

    bm25_rels = assign_relevance(bm25_papers[:k], relevant_ids)
    ce_rels = assign_relevance(reranked_papers[:k], relevant_ids)

    return {
        "bm25_ndcg": round(ndcg_at_k(bm25_rels, k), 4),
        "bm25_mrr": round(mrr(bm25_rels), 4),
        "ce_ndcg": round(ndcg_at_k(ce_rels, k), 4),
        "ce_mrr": round(mrr(ce_rels), 4),
    }


def print_results(query: str, bm25_papers: list[dict], reranked_papers: list[dict],
                  metrics: dict, k: int = 10) -> None:
    """Pretty-print the comparison table and metrics to stdout."""
    print("=" * 80)
    print(f"Query: {query!r}")
    print("=" * 80)

    table = build_comparison_table(bm25_papers, reranked_papers, top_k=k)
    print(table.to_string())

    print()
    print("-" * 40)
    print(f"{'Metric':<20} {'BM25':>10} {'CrossEncoder':>14}")
    print("-" * 40)
    print(f"{'NDCG@' + str(k):<20} {metrics['bm25_ndcg']:>10.4f} {metrics['ce_ndcg']:>14.4f}")
    print(f"{'MRR':<20} {metrics['bm25_mrr']:>10.4f} {metrics['ce_mrr']:>14.4f}")
    print("-" * 40)


if __name__ == "__main__":
    import argparse
    from retriever import retrieve
    from reranker import load_model, rerank

    parser = argparse.ArgumentParser(description="Evaluate BM25 vs CrossEncoder reranking")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--limit", type=int, default=50, help="Candidates to fetch")
    parser.add_argument("--k", type=int, default=10, help="Evaluation cutoff (NDCG@k, top-k display)")
    parser.add_argument("--relevant-ids", nargs="*", default=None,
                        help="Space-separated Semantic Scholar paper IDs to treat as relevant")
    args = parser.parse_args()

    bm25_papers = retrieve(args.query, limit=args.limit)
    model = load_model()
    reranked_papers = rerank(args.query, bm25_papers, model)

    relevant_ids = set(args.relevant_ids) if args.relevant_ids else None
    metrics = evaluate(bm25_papers, reranked_papers, relevant_ids=relevant_ids, k=args.k)
    print_results(args.query, bm25_papers, reranked_papers, metrics, k=args.k)
