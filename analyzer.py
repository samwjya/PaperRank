"""LLM-powered analysis: question decomposition, per-paper coverage, gaps, summaries.

All LLM calls go through Groq. Abstracts are untrusted text, so every quote the
model returns as evidence is checked against the abstract before we trust it.
"""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from groq import Groq

from retriever import count_works


# Qwen kept the question's core subject as an aspect and gave stable output
# across runs; gpt-oss-120b sometimes dropped it. Override with GROQ_MODEL.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
# gpt-oss is a reasoning model: its hidden reasoning tokens count toward
# max_tokens, so if selected we keep reasoning short and add headroom.
REASONING_HEADROOM = 1000
MAX_ABSTRACT_CHARS = 2500   # keeps prompts inside Groq free-tier token limits
PAPERS_PER_CALL = 5         # coverage is checked in batches of this many papers

STATUS_POINTS = {"covered": 1.0, "partial": 0.5, "missing": 0.0}


class LLMError(RuntimeError):
    """Raised when the LLM is unavailable or returns something unusable."""


_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise LLMError("GROQ_API_KEY is not configured.")
        _client = Groq(api_key=api_key, max_retries=2, timeout=60)
    return _client


def _chat(prompt: str, max_tokens: int, as_json: bool = False) -> str:
    kwargs = {"response_format": {"type": "json_object"}} if as_json else {}
    if "gpt-oss" in GROQ_MODEL:
        kwargs["reasoning_effort"] = "low"
        max_tokens += REASONING_HEADROOM
    try:
        resp = _get_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
            **kwargs,
        )
    except LLMError:
        raise
    except Exception as e:
        raise LLMError(f"LLM request failed: {e}") from e
    return resp.choices[0].message.content or ""


def chat_json(prompt: str, max_tokens: int) -> dict:
    raw = _chat(prompt, max_tokens, as_json=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise LLMError("LLM returned invalid JSON.") from e


def _normalize(text: str) -> str:
    # Lowercase, unify hyphen variants, collapse whitespace — so a quote still
    # matches when the model changes "‐" to "-" or reflows spaces.
    text = re.sub(r"[‐-―−]", "-", text.lower())
    return re.sub(r"\s+", " ", text).strip()


# ── 1. Decomposition ──────────────────────────────────────────────────────────

def decompose(question: str) -> dict:
    """Split a research question into aspects and search sub-queries.

    Returns {"aspects": [{"name", "keywords"}, ...], "sub_queries": [str, ...]}.
    If the LLM fails, falls back to a single sub-query (the question itself)
    and no aspects, so search still works without coverage analysis.
    """
    prompt = f"""You help researchers search the academic literature.

Research question: {question}

1. Break the question into 2-5 distinct ASPECTS a relevant paper should address
   (e.g. the core subject, the method, the target/organism, the condition,
   the setting). Put the question's main topic first, named after the topic itself.
   Alternatives joined by "or" form ONE aspect. Skip generic words that no
   paper could fail to cover ("effects", "methods", "recent", "papers").
   Stay at the question's own level of generality: if it says "delivery
   methods" the aspect is delivery methods, not one specific method; if it
   says "liver diseases", not one specific disease.
   For each, give a short plain-English name saying what it is (e.g. "Target organ",
   "Gene transfer"; never a generic label like "Core subject") and ONE
   search KEYWORD: 1-3 normal words, taken from the question's wording where
   possible (e.g. "delivery", "liver disease", "in vivo"). Never a list,
   no commas, no boolean operators.
2. Write 3-4 SUB-QUERIES for semantic search. Each combines several aspects
   in different ways or uses different terminology, so together they reach
   different parts of the literature. Never search a single broad aspect alone.

Respond with JSON only:
{{"aspects": [{{"name": "...", "keyword": "..."}}],
  "sub_queries": ["...", "..."]}}"""
    try:
        data = chat_json(prompt, max_tokens=500)
        aspects = [
            # Keep only the first term if the model still returns a list: counts
            # AND all keywords together, so a list would match almost nothing.
            {"name": str(a["name"]).strip(),
             "keywords": re.split(r"[,;/]| or ", str(a.get("keyword") or a["name"]))[0].strip()}
            for a in data.get("aspects", [])
            if a.get("name")
        ][:5]
        sub_queries = [str(q).strip() for q in data.get("sub_queries", []) if str(q).strip()][:4]
    except (LLMError, KeyError, TypeError, AttributeError):
        return {"aspects": [], "sub_queries": [question]}
    return {"aspects": aspects, "sub_queries": sub_queries or [question]}


# ── 2. Coverage ───────────────────────────────────────────────────────────────

def _assess_batch(question: str, aspect_names: list[str], papers: list[dict]) -> list[dict]:
    papers_text = "\n\n".join(
        f"[{i}] Title: {p['title']}\nAbstract: {p['abstract'][:MAX_ABSTRACT_CHARS]}"
        for i, p in enumerate(papers)
    )
    prompt = f"""Research question: {question}
Aspects: {json.dumps(aspect_names)}

For each paper below, judge each aspect using ONLY its title and abstract:
- "covered": the paper clearly addresses it
- "partial": touched on, indirectly or only in part
- "missing": not addressed
For covered/partial, "evidence" must be a short EXACT quote copied from the
abstract (max 25 words). For missing, evidence is "".
Also add "note": one short sentence on what the paper does NOT address
relative to the question ("" if nothing important is missing).

{papers_text}

Respond with JSON only:
{{"papers": [{{"index": 0, "aspects": [{{"name": "...", "status": "...", "evidence": "..."}}], "note": "..."}}]}}"""
    data = chat_json(prompt, max_tokens=350 * len(papers))
    by_index = {r.get("index"): r for r in data.get("papers", []) if isinstance(r, dict)}

    results = []
    for i, paper in enumerate(papers):
        row = by_index.get(i, {})
        given = {a.get("name"): a for a in row.get("aspects", []) if isinstance(a, dict)}
        abstract_norm = _normalize(paper["title"] + " " + paper["abstract"])
        aspects = []
        for name in aspect_names:
            a = given.get(name, {})
            status = a.get("status") if a.get("status") in STATUS_POINTS else "missing"
            evidence = str(a.get("evidence") or "").strip().strip('"')
            # Trust a "covered" verdict only if its quote really is in the abstract.
            if evidence and _normalize(evidence) not in abstract_norm:
                evidence = ""
                if status == "covered":
                    status = "partial"
            aspects.append({"name": name, "status": status, "evidence": evidence})
        score = sum(STATUS_POINTS[a["status"]] for a in aspects) / len(aspects)
        results.append({
            "aspects": aspects,
            "coverage": round(score * 100),
            "note": str(row.get("note") or "").strip(),
        })
    return results


def assess_coverage(question: str, aspects: list[dict], papers: list[dict]) -> list[dict]:
    """Check which aspects each paper covers.

    Returns one dict per paper, in the same order:
    {"aspects": [{"name", "status", "evidence"}], "coverage": 0-100, "note": str}.
    Batches run in parallel; raises LLMError if any batch fails.
    """
    if not aspects or not papers:
        return [{"aspects": [], "coverage": None, "note": ""} for _ in papers]
    names = [a["name"] for a in aspects]
    batches = [papers[i:i + PAPERS_PER_CALL] for i in range(0, len(papers), PAPERS_PER_CALL)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        batch_results = list(pool.map(lambda b: _assess_batch(question, names, b), batches))
    return [r for batch in batch_results for r in batch]


# ── 3. Gaps ───────────────────────────────────────────────────────────────────

def find_gaps(aspects: list[dict], assessments: list[dict]) -> dict:
    """Find aspects the literature seems thin on.

    Two signals:
    - `uncovered`: aspects that no assessed paper fully covers.
    - `bottlenecks`: leave-one-out keyword counts. If all aspects together match
      7 papers but dropping aspect X gives 640, X is what makes the combination rare.
    These are leads to verify, not proof (wording differences, abstracts only).
    """
    uncovered = [
        a["name"] for i, a in enumerate(aspects)
        if not any(r["aspects"] and r["aspects"][i]["status"] == "covered" for r in assessments)
    ]
    result = {"uncovered": uncovered, "full_count": None, "full_query": "", "bottlenecks": []}
    if len(aspects) < 2:
        return result

    full_query = " ".join(a["keywords"] for a in aspects)
    leave_one_out = [
        " ".join(b["keywords"] for b in aspects if b is not a) for a in aspects
    ]
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            counts = list(pool.map(count_works, [full_query, *leave_one_out]))
    except Exception:
        return result  # counts are a bonus; coverage-based gaps still stand

    full_count, rest = counts[0], counts[1:]
    result["full_count"] = full_count
    result["full_query"] = full_query
    result["bottlenecks"] = sorted(
        (
            {"aspect": a["name"], "query_without": q, "count_without": c}
            for a, q, c in zip(aspects, leave_one_out, rest)
            # Only interesting if dropping this aspect opens up much more literature.
            if c >= max(3 * full_count, full_count + 20)
        ),
        key=lambda b: -b["count_without"],
    )
    return result


# ── 4. Summaries ──────────────────────────────────────────────────────────────

def summarize_paper(question: str, paper: dict) -> str:
    """2-3 sentence summary of the paper's contribution, focused on the question."""
    prompt = f"""A researcher is asking: {question}

Summarize this paper in 2-3 sentences: its key contribution and findings,
and what it tells the researcher about their question. Use only the abstract.

Title: {paper['title']}
Abstract: {paper['abstract'][:MAX_ABSTRACT_CHARS]}"""
    return _chat(prompt, max_tokens=200).strip()

