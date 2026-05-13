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


def auto_relevance(papers: list[dict], top_n: int = 10) -> set[str]:
    """Derive a pseudo-relevance set from the top-N OpenAlex results.

    In the absence of manual judgments, we treat the top-N OpenAlex papers as
    relevant. This is a recall-based approximation — useful for demo and
    relative comparison, not absolute metric benchmarking.
    """
    openalex_sorted = sorted(papers, key=lambda p: p.get("openalex_rank", 9999))
    return {p["paperId"] for p in openalex_sorted[:top_n] if p.get("paperId")}


def build_comparison_table(baseline_papers: list[dict], reranked_papers: list[dict], top_k: int = 10) -> pd.DataFrame:
    """Build a side-by-side DataFrame comparing OpenAlex baseline and ensemble top-K results."""
    rows = []
    for i in range(top_k):
        base_p = baseline_papers[i] if i < len(baseline_papers) else {}
        ce_p = reranked_papers[i] if i < len(reranked_papers) else {}

        base_title = (base_p.get("title") or "")[:60]
        ce_title = (ce_p.get("title") or "")[:60]

        rows.append({
            "Rank": i + 1,
            "OpenAlex Title": base_title,
            "OpenAlex Rank": base_p.get("openalex_rank", ""),
            "Ensemble Title": ce_title,
            "Ensemble Score": ce_p.get("ce_score", ""),
        })
    return pd.DataFrame(rows).set_index("Rank")


def evaluate(baseline_papers: list[dict], reranked_papers: list[dict],
             relevant_ids: set[str] | None = None, k: int = 10) -> dict:
    """Compute NDCG@k and MRR for both OpenAlex baseline and ensemble reranked lists.

    If relevant_ids is None, falls back to auto_relevance (pseudo-relevance
    from OpenAlex top-k). Returns a dict with all four metrics.
    """
    if relevant_ids is None:
        relevant_ids = auto_relevance(baseline_papers, top_n=k)

    openalex_rels = assign_relevance(baseline_papers[:k], relevant_ids)
    ensemble_rels = assign_relevance(reranked_papers[:k], relevant_ids)

    return {
        "openalex_ndcg": round(ndcg_at_k(openalex_rels, k), 4),
        "openalex_mrr": round(mrr(openalex_rels), 4),
        "ensemble_ndcg": round(ndcg_at_k(ensemble_rels, k), 4),
        "ensemble_mrr": round(mrr(ensemble_rels), 4),
    }


def print_results(query: str, baseline_papers: list[dict], reranked_papers: list[dict],
                  metrics: dict, k: int = 10, ensemble: bool = False) -> None:
    """Pretty-print the comparison table and metrics to stdout."""
    print("=" * 80)
    print(f"Query: {query!r}")
    print("=" * 80)

    table = build_comparison_table(baseline_papers, reranked_papers, top_k=k)
    print(table.to_string())

    print()
    print("-" * 40)
    print(f"{'Metric':<20} {'OpenAlex':>10} {'Ensemble':>14}")
    print("-" * 40)
    print(f"{'NDCG@' + str(k):<20} {metrics['openalex_ndcg']:>10.4f} {metrics['ensemble_ndcg']:>14.4f}")
    print(f"{'MRR':<20} {metrics['openalex_mrr']:>10.4f} {metrics['ensemble_mrr']:>14.4f}")
    print("-" * 40)

    print()
    print("OpenAlex Links:")
    for i, p in enumerate(baseline_papers[:k], start=1):
        url = p.get("url") or "N/A"
        print(f"  {i:>2}. {p.get('title', '')[:70]}  —  {url}")

    print()
    print("Ensemble Links:")
    for i, p in enumerate(reranked_papers[:k], start=1):
        url = p.get("url") or "N/A"
        print(f"  {i:>2}. {p.get('title', '')[:70]}  —  {url}")

    if ensemble:
        print()
        header = f"  {'#':<4} {'Title':<48} {'BGE-M3':>8} {'MiniLM':>8} {'Electra':>8} {'Ensemble':>10}"
        print(f"Ensemble Score Breakdown (top-{k} reranked):")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for p in reranked_papers[:k]:
            title = (p.get("title") or "")[:46]
            s1 = p.get("ce_score_model1", "N/A")
            s2 = p.get("ce_score_model2", "N/A")
            s3 = p.get("ce_score_model3", "N/A")
            ens = p.get("ce_score_ensemble", "N/A")
            print(f"  {p['ce_rank']:<4} {title:<48} {s1!s:>8} {s2!s:>8} {s3!s:>8} {ens!s:>10}")


if __name__ == "__main__":
    import argparse
    from retriever import fetch_papers
    from reranker import load_model, rerank

    parser = argparse.ArgumentParser(description="Evaluate OpenAlex baseline vs ensemble reranking")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--limit", type=int, default=50, help="Candidates to fetch")
    parser.add_argument("--k", type=int, default=10, help="Evaluation cutoff (NDCG@k, top-k display)")
    parser.add_argument("--relevant-ids", nargs="*", default=None,
                        help="Space-separated OpenAlex paper IDs to treat as relevant")
    args = parser.parse_args()

    baseline_papers = fetch_papers(args.query, limit=args.limit)
    model = load_model()
    reranked_papers = rerank(args.query, baseline_papers, model)

    relevant_ids = set(args.relevant_ids) if args.relevant_ids else None
    metrics = evaluate(baseline_papers, reranked_papers, relevant_ids=relevant_ids, k=args.k)
    print_results(args.query, baseline_papers, reranked_papers, metrics, k=args.k)
