from sentence_transformers import CrossEncoder


DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

AVAILABLE_MODELS = [
    "BAAI/bge-reranker-v2-m3",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-electra-base",
]


def load_model(model_name: str = DEFAULT_MODEL) -> CrossEncoder:
    """Load a CrossEncoder model from HuggingFace (cached after first download)."""
    print(f"Loading reranker model: {model_name} ...")
    model = CrossEncoder(model_name, max_length=512, trust_remote_code=True)
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


def rerank(query: str, papers: list[dict], model: CrossEncoder, batch_size: int = 32) -> list[dict]:
    """Score every (query, paper) pair with the CrossEncoder and return papers sorted by score.

    Returns the same paper dicts, each annotated with `ce_score` and `ce_rank`,
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
