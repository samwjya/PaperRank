import requests
from rank_bm25 import BM25Okapi


SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,abstract,year,authors,citationCount"


def fetch_papers(query: str, limit: int = 50) -> list[dict]:
    """Fetch candidate papers from the Semantic Scholar API."""
    params = {
        "query": query,
        "limit": limit,
        "fields": FIELDS,
    }
    response = requests.get(SEMANTIC_SCHOLAR_URL, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    papers = data.get("data", [])
    # Drop papers with no abstract — BM25 and reranker both need text
    return [p for p in papers if p.get("abstract")]


def tokenize(text: str) -> list[str]:
    """Lowercase and whitespace-split text into tokens for BM25."""
    return text.lower().split()


def bm25_rank(query: str, papers: list[dict]) -> list[dict]:
    """Rank papers against the query using BM25Okapi."""
    corpus = [tokenize((p.get("title") or "") + " " + (p.get("abstract") or ""))
              for p in papers]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(query))

    ranked = sorted(
        zip(papers, scores),
        key=lambda x: x[1],
        reverse=True,
    )
    results = []
    for rank, (paper, score) in enumerate(ranked, start=1):
        paper = dict(paper)
        paper["bm25_score"] = round(float(score), 4)
        paper["bm25_rank"] = rank
        results.append(paper)
    return results


def retrieve(query: str, limit: int = 50) -> list[dict]:
    """Full retrieval pipeline: fetch from Semantic Scholar then BM25-rank."""
    papers = fetch_papers(query, limit=limit)
    if not papers:
        raise ValueError(f"No papers with abstracts found for query: {query!r}")
    ranked = bm25_rank(query, papers)
    return ranked


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrieve and BM25-rank papers from Semantic Scholar")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--limit", type=int, default=50, help="Number of candidates to fetch")
    parser.add_argument("--top", type=int, default=10, help="Number of results to display")
    args = parser.parse_args()

    results = retrieve(args.query, limit=args.limit)

    print(f"\nQuery: {args.query!r}")
    print(f"Fetched {len(results)} papers, showing top {args.top}:\n")
    for paper in results[:args.top]:
        authors = ", ".join(a["name"] for a in paper.get("authors", [])[:2])
        print(f"[BM25 #{paper['bm25_rank']}] score={paper['bm25_score']}")
        print(f"  Title : {paper['title']}")
        print(f"  Year  : {paper.get('year', 'N/A')}  |  Citations: {paper.get('citationCount', 'N/A')}")
        print(f"  Authors: {authors}")
        print()
