import torch
from sentence_transformers import CrossEncoder


DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

AVAILABLE_MODELS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "BAAI/bge-reranker-v2-m3",
    "cross-encoder/ms-marco-electra-base",
]

# (label, minimum score). Checked top-down; first match wins.
TIERS = [
    ("Perfectly Relevant", 0.90),
    ("Relevant",           0.70),
    ("Somewhat Relevant",  0.50),
    ("Not Relevant",       float("-inf")),
]


def load_model(model_name: str = DEFAULT_MODEL) -> CrossEncoder:
    """Load a CrossEncoder model from HuggingFace (cached on disk after first download).

    Sigmoid is forced so every model returns scores in [0, 1]. Without it,
    the ms-marco models return raw logits (their saved config uses Identity)
    while BGE returns probabilities, and the tier thresholds stop meaning anything.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model {model_name!r}. Choose from: {AVAILABLE_MODELS}")
    return CrossEncoder(model_name, max_length=512, activation_fn=torch.nn.Sigmoid())


def relevance_tier(score: float) -> str:
    """Map a [0, 1] CrossEncoder score to a relevance tier label."""
    for label, threshold in TIERS:
        if score >= threshold:
            return label
    return TIERS[-1][0]


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

    Returns copies of the paper dicts, each annotated with `ce_score`,
    `ce_rank` and `tier`, sorted by CrossEncoder score descending.
    """
    if not papers:
        return []
    pairs = build_pairs(query, papers)
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

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
        paper["tier"] = relevance_tier(paper["ce_score"])
        results.append(paper)
    return results


if __name__ == "__main__":
    import argparse
    import sys
    from retriever import fetch_papers

    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

    parser = argparse.ArgumentParser(description="Rerank OpenAlex candidates with a CrossEncoder")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, choices=AVAILABLE_MODELS)
    parser.add_argument("--limit", type=int, default=50, help="Number of OpenAlex candidates to fetch")
    parser.add_argument("--top", type=int, default=10, help="Number of reranked results to display")
    args = parser.parse_args()

    papers = fetch_papers(args.query, limit=args.limit)
    print(f"Loading {args.model} ...")
    model = load_model(args.model)
    reranked = rerank(args.query, papers, model)

    print(f"\nQuery: {args.query!r}")
    print(f"Showing top {args.top} after CrossEncoder reranking:\n")
    for paper in reranked[:args.top]:
        authors = ", ".join(a["name"] for a in paper.get("authors", [])[:2])
        print(f"[CE #{paper['ce_rank']}] {paper['ce_score']:.3f} {paper['tier']}  (was OpenAlex #{paper['openalex_rank']})")
        print(f"  Title : {paper['title']}")
        print(f"  Year  : {paper.get('year', 'N/A')}  |  Citations: {paper.get('citationCount', 'N/A')}")
        print(f"  Authors: {authors}")
        print()
