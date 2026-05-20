import os
import requests


OPEN_ALEX_URL = "https://api.openalex.org/works"


def _load_api_key() -> str:
    return os.environ.get("OPEN_ALEX_API", "")

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
    """Fetch papers from OpenAlex and return them in API relevance order."""
    params = {
        "search.semantic": query,
        "per_page": min(limit, 200),
        "api_key": _load_api_key(),
    }
    response = requests.get(OPEN_ALEX_URL, params=params, timeout=15)
    response.raise_for_status()
    raw_papers = response.json().get("results", [])

    papers = []
    openalex_rank = 1
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
            "openalex_rank": openalex_rank,
        })
        openalex_rank += 1

    if not papers:
        raise ValueError(f"No papers with abstracts found for query: {query!r}")
    return papers


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch papers from OpenAlex in relevance order")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--limit", type=int, default=50, help="Number of candidates to fetch")
    parser.add_argument("--top", type=int, default=10, help="Number of results to display")
    args = parser.parse_args()

    results = fetch_papers(args.query, limit=args.limit)

    print(f"\nQuery: {args.query!r}")
    print(f"Fetched {len(results)} papers, showing top {args.top}:\n")
    for paper in results[:args.top]:
        authors = ", ".join(a["name"] for a in paper.get("authors", [])[:2])
        print(f"[OpenAlex #{paper['openalex_rank']}]")
        print(f"  Title  : {paper['title']}")
        print(f"  Year   : {paper.get('year', 'N/A')}  |  Citations: {paper.get('citationCount', 'N/A')}")
        print(f"  Authors: {authors}")
        print()
