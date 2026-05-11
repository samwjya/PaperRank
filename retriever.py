import os
import requests
from rank_bm25 import BM25Okapi


OPEN_ALEX_URL = "https://api.openalex.org/works"


def _load_api_key() -> str:
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    with open(env_path) as f:
        for line in f:
            if line.startswith("OPEN_ALEX_API="):
                return line.strip().split("=", 1)[1]
    return ""


def _reconstruct_abstract(inverted_index: dict) -> str:
    """OpenAlex stores abstracts as an inverted index; reconstruct plain text."""
    if not inverted_index:
        return ""
    max_pos = max(pos for positions in inverted_index.values() for pos in positions)
    words = [""] * (max_pos + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words)


def fetch_papers(query: str, limit: int = 50) -> list[dict]:
    """Fetch candidate papers from the OpenAlex API."""
    params = {
        "search.semantic": query,
        "per_page": min(limit, 200),
        "api_key": _load_api_key(),
    }
    response = requests.get(OPEN_ALEX_URL, params=params, timeout=15)
    response.raise_for_status()
    raw_papers = response.json().get("results", [])

    papers = []
    for p in raw_papers:
        abstract = _reconstruct_abstract(p.get("abstract_inverted_index") or {})
        if not abstract:
            continue
        doi = p.get("doi") or ""
        openalex_id = p.get("id") or ""
        url = doi if doi else openalex_id
        papers.append({
            "paperId": openalex_id,
            "title": p.get("title") or "",
            "abstract": abstract,
            "year": p.get("publication_year"),
            "citationCount": p.get("cited_by_count"),
            "url": url,
            "authors": [
                {"name": a["author"]["display_name"]}
                for a in p.get("authorships", [])
                if a.get("author") and a["author"].get("display_name")
            ],
        })
    return papers


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
