import os
import re
from concurrent.futures import ThreadPoolExecutor

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


OPEN_ALEX_URL = "https://api.openalex.org/works"

# OpenAlex hard limits: semantic search returns at most 50 results and has no
# second page; keyword search allows up to 200 per page.
SEMANTIC_MAX = 50
KEYWORD_MAX = 200

# Only the fields we use — keeps responses small.
_SELECT = "id,doi,title,publication_year,cited_by_count,authorships,abstract_inverted_index"


def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "PaperRank/1.0"
    return session


_session = _make_session()


def _base_params() -> dict:
    # Every request costs against a daily budget; a key gives its own $1/day
    # instead of sharing the anonymous per-IP budget.
    key = os.environ.get("OPEN_ALEX_API", "")
    return {"api_key": key} if key else {}


def _reconstruct_abstract(inverted_index: dict) -> str:
    """OpenAlex stores abstracts as an inverted index; reconstruct plain text."""
    if not inverted_index:
        return ""
    max_pos = max(pos for positions in inverted_index.values() for pos in positions)
    words = [""] * (max_pos + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(w for w in words if w)


def _parse_paper(p: dict) -> dict | None:
    abstract = _reconstruct_abstract(p.get("abstract_inverted_index") or {})
    title = p.get("title") or ""
    if not abstract or not title:
        return None
    openalex_id = p.get("id") or ""
    return {
        "paperId": openalex_id,
        "title": title,
        "abstract": abstract,
        "year": p.get("publication_year"),
        "citationCount": p.get("cited_by_count"),
        "url": p.get("doi") or openalex_id,
        "authors": [
            {"name": a["author"]["display_name"]}
            for a in p.get("authorships", [])
            if a.get("author") and a["author"].get("display_name")
        ],
    }


def fetch_papers(query: str, limit: int = 50, mode: str = "semantic") -> list[dict]:
    """Fetch papers from OpenAlex in its relevance order.

    mode="semantic" uses embedding search (max 50); mode="keyword" uses
    full-text keyword search (max 200). Papers without an abstract are dropped,
    so fewer than `limit` may come back. Returns [] if nothing matches.
    Each paper gets `openalex_rank` (1-based, after dropping abstract-less ones).
    """
    if mode == "semantic":
        params = {"search.semantic": query, "per_page": min(limit, SEMANTIC_MAX)}
    elif mode == "keyword":
        params = {"search": query, "per_page": min(limit, KEYWORD_MAX)}
    else:
        raise ValueError(f"Unknown mode {mode!r}; use 'semantic' or 'keyword'")

    params |= {"select": _SELECT, **_base_params()}
    response = _session.get(OPEN_ALEX_URL, params=params, timeout=20)
    response.raise_for_status()

    papers = []
    for raw in response.json().get("results", []):
        paper = _parse_paper(raw)
        if paper:
            paper["openalex_rank"] = len(papers) + 1
            papers.append(paper)
    return papers


def multi_search(queries: list[str], limit_each: int = 50, mode: str = "semantic") -> list[dict]:
    """Run several searches in parallel and merge them, removing duplicates.

    Each merged paper records `matched_queries` (which sub-queries found it) and
    `openalex_rank` (its best rank across those searches). Order: papers found by
    more sub-queries first, then by best rank — the reranker reorders anyway.
    A sub-query that fails is skipped rather than failing the whole search.
    """
    def run(q: str) -> list[dict]:
        try:
            return fetch_papers(q, limit=limit_each, mode=mode)
        except requests.RequestException:
            return []

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, queries))

    merged: dict[str, dict] = {}
    for query, papers in zip(queries, results):
        for paper in papers:
            pid = paper["paperId"]
            if pid not in merged:
                merged[pid] = {**paper, "matched_queries": [query]}
            else:
                entry = merged[pid]
                entry["matched_queries"].append(query)
                entry["openalex_rank"] = min(entry["openalex_rank"], paper["openalex_rank"])

    return sorted(merged.values(), key=lambda p: (-len(p["matched_queries"]), p["openalex_rank"]))


def count_works(query: str) -> int:
    """Count works whose title or abstract match `query` (keyword match).

    Used as a rough "how studied is this combination?" signal for gap detection.
    Title+abstract is far more specific than full-text search: in testing a
    4-term query matched 7 works this way versus 6,671 via full text.
    """
    # Commas and pipes are filter-syntax separators in OpenAlex.
    clean = re.sub(r"[,|:]", " ", query).strip()
    params = {
        "filter": f"title_and_abstract.search:{clean}",
        "per_page": 1,
        "select": "id",
        **_base_params(),
    }
    response = _session.get(OPEN_ALEX_URL, params=params, timeout=20)
    response.raise_for_status()
    return response.json()["meta"]["count"]


if __name__ == "__main__":
    import argparse
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

    parser = argparse.ArgumentParser(description="Fetch papers from OpenAlex in relevance order")
    parser.add_argument("queries", nargs="+", help="One or more search queries (several = multi-search)")
    parser.add_argument("--limit", type=int, default=50, help="Candidates to fetch per query")
    parser.add_argument("--mode", choices=["semantic", "keyword"], default="semantic")
    parser.add_argument("--top", type=int, default=5, help="Number of results to display")
    args = parser.parse_args()

    if len(args.queries) == 1:
        results = fetch_papers(args.queries[0], limit=args.limit, mode=args.mode)
    else:
        results = multi_search(args.queries, limit_each=args.limit, mode=args.mode)

    print(f"\nFetched {len(results)} unique papers with abstracts, showing top {args.top}:\n")
    for paper in results[:args.top]:
        authors = ", ".join(a["name"] for a in paper.get("authors", [])[:2])
        found_by = len(paper.get("matched_queries", [args.queries[0]]))
        print(f"[OpenAlex best #{paper['openalex_rank']}, found by {found_by} quer{'y' if found_by == 1 else 'ies'}]")
        print(f"  Title  : {paper['title']}")
        print(f"  Year   : {paper.get('year', 'N/A')}  |  Citations: {paper.get('citationCount', 'N/A')}")
        print(f"  Authors: {authors}")
        print()

    for q in args.queries:
        print(f"title/abstract matches for {q!r}: {count_works(q)}")
