from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import CrossEncoder


DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

ENSEMBLE_MODELS = [
    "BAAI/bge-reranker-v2-m3",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-electra-base",
]


def load_model(model_name: str = DEFAULT_MODEL) -> CrossEncoder:
    """Load a CrossEncoder model from HuggingFace (cached after first download)."""
    print(f"Loading reranker model: {model_name} ...")
    model = CrossEncoder(model_name, max_length=512)
    print("Model loaded.\n")
    return model


def build_pairs(query: str, papers: list[dict]) -> list[tuple[str, str]]:
    """Build (query, document) pairs for CrossEncoder scoring.

    The document is title + abstract concatenated — both fields carry
    relevance signal and the CrossEncoder attends over the full pair.
    """
    pairs = []
    for paper in papers:
        title = paper.get("title") or ""
        abstract = paper.get("abstract") or ""
        document = f"{title}. {abstract}".strip()
        pairs.append((query, document))
    return pairs


def rerank(query: str, papers: list[dict], model: CrossEncoder, batch_size: int = 16) -> list[dict]:
    """Score every (query, paper) pair with the CrossEncoder and return papers sorted by score.

    Args:
        query:      The original search query string.
        papers:     List of paper dicts (already BM25-ranked; order does not matter here).
        model:      A loaded CrossEncoder instance.
        batch_size: Number of pairs to score in one forward pass.

    Returns:
        The same paper dicts, each annotated with `ce_score` and `ce_rank`,
        sorted by CrossEncoder score descending.
    """
    pairs = build_pairs(query, papers)
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=True)

    ranked = sorted(
        zip(papers, scores),
        key=lambda x: x[1],
        reverse=True,
    )
    results = []
    for rank, (paper, score) in enumerate(ranked, start=1):
        paper = dict(paper)
        paper["ce_score"] = round(float(score), 4)
        paper["ce_rank"] = rank
        results.append(paper)
    return results


def load_models(model_names: list[str] = ENSEMBLE_MODELS) -> list[CrossEncoder]:
    """Load all CrossEncoder models in parallel using ThreadPoolExecutor."""
    with ThreadPoolExecutor(max_workers=len(model_names)) as executor:
        futures = [executor.submit(load_model, name) for name in model_names]
        return [f.result() for f in futures]


def _normalize(scores: list[float]) -> list[float]:
    """Min-max normalize scores to [0, 1]. Returns all 1.0 when all scores are equal."""
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def ensemble_rerank(
    query: str,
    papers: list[dict],
    models: list[CrossEncoder],
    batch_size: int = 16,
) -> list[dict]:
    """Score papers with multiple CrossEncoders, normalize each model's scores to
    [0, 1] via min-max, then average across models (equal weights) to produce an
    ensemble score. Returns papers sorted by ensemble score descending.

    Each returned paper dict gains:
      ce_score_model1 / 2 / 3  — raw score from each model
      ce_score_ensemble         — mean of the three normalized scores
      ce_score                  — alias for ce_score_ensemble (pipeline compat)
      ce_rank                   — 1-based position in the ensemble ranking
    """
    pairs = build_pairs(query, papers)

    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = [executor.submit(m.predict, pairs) for m in models]
        raw_scores = [list(f.result()) for f in futures]
    norm_scores = [_normalize(s) for s in raw_scores]

    n_models = len(models)
    ensemble = [
        sum(norm_scores[m][i] for m in range(n_models)) / n_models
        for i in range(len(papers))
    ]

    order = sorted(range(len(papers)), key=lambda i: ensemble[i], reverse=True)

    results = []
    for rank, i in enumerate(order, start=1):
        paper = dict(papers[i])
        for m_idx, raw in enumerate(raw_scores, start=1):
            paper[f"ce_score_model{m_idx}"] = round(float(raw[i]), 4)
        paper["ce_score_ensemble"] = round(float(ensemble[i]), 4)
        paper["ce_score"] = paper["ce_score_ensemble"]
        paper["ce_rank"] = rank
        results.append(paper)
    return results


if __name__ == "__main__":
    import argparse
    from retriever import fetch_papers

    parser = argparse.ArgumentParser(description="Rerank OpenAlex candidates with a CrossEncoder")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HuggingFace CrossEncoder model ID")
    parser.add_argument("--limit", type=int, default=50, help="Number of OpenAlex candidates to fetch")
    parser.add_argument("--top", type=int, default=10, help="Number of reranked results to display")
    args = parser.parse_args()

    papers = fetch_papers(args.query, limit=args.limit)
    model = load_model(args.model)
    reranked = rerank(args.query, papers, model)

    print(f"\nQuery: {args.query!r}")
    print(f"Showing top {args.top} after CrossEncoder reranking:\n")
    for paper in reranked[:args.top]:
        authors = ", ".join(a["name"] for a in paper.get("authors", [])[:2])
        print(f"[CE #{paper['ce_rank']}] ce_score={paper['ce_score']}  (was OpenAlex #{paper['openalex_rank']})")
        print(f"  Title : {paper['title']}")
        print(f"  Year  : {paper.get('year', 'N/A')}  |  Citations: {paper.get('citationCount', 'N/A')}")
        print(f"  Authors: {authors}")
        print()
