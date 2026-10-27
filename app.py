import os
import re
import threading
import time
from datetime import date

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from analyzer import LLMError, assess_coverage, decompose, find_gaps, summarize_paper
from reranker import AVAILABLE_MODELS, DEFAULT_MODEL, load_model, rerank
from retriever import multi_search


MAX_QUESTION_CHARS = 500
CACHE_TTL = 24 * 3600
# OpenAlex bills ~$0.009 per search against a $1/day key budget, so cap the
# whole app; per-session caps stop one visitor from using it all.
DAILY_SEARCH_LIMIT = int(os.environ.get("DAILY_SEARCH_LIMIT", "100"))
SESSION_SEARCHES_PER_HOUR = 10
SESSION_SUMMARIES_PER_HOUR = 30

MODEL_LABELS = {
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "MiniLM — fast",
    "BAAI/bge-reranker-v2-m3": "BGE v2 m3 — most accurate, slower",
    "cross-encoder/ms-marco-electra-base": "Electra — medium",
}
STATUS_ICON = {"covered": "✅", "partial": "🟡", "missing": "❌"}

st.set_page_config(page_title="PaperRank", page_icon="📄", layout="wide")


class NoResults(Exception):
    pass


# ── Cached pipeline steps ─────────────────────────────────────────────────────
# Each step raises instead of returning a fallback on failure, so a failure is
# never cached for 24h; the caller handles the fallback.

@st.cache_resource(show_spinner=False)
def get_model(name: str):
    return load_model(name)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_plan(question: str) -> dict:
    plan = decompose(question)
    if not plan["aspects"]:
        raise LLMError("Could not break the question into aspects.")
    return plan


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_search(sub_queries: tuple[str, ...]) -> list[dict]:
    papers = multi_search(list(sub_queries))
    if not papers:
        raise NoResults()
    return papers


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_rerank(question: str, sub_queries: tuple[str, ...], model_name: str) -> list[dict]:
    return rerank(question, cached_search(sub_queries), get_model(model_name))


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_assess(question: str, aspects: list[dict], papers: list[dict]) -> list[dict]:
    return assess_coverage(question, aspects, papers)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_gaps(aspects: list[dict], assessments: list[dict]) -> dict:
    return find_gaps(aspects, assessments)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_summary(question: str, paper_id: str, title: str, abstract: str) -> str:
    return summarize_paper(question, {"title": title, "abstract": abstract})


# ── Rate limiting ─────────────────────────────────────────────────────────────

@st.cache_resource
def _daily_usage() -> dict:
    # One shared counter for the whole server process (all visitors).
    return {"lock": threading.Lock(), "day": "", "count": 0}


def _take_daily_search() -> bool:
    usage = _daily_usage()
    with usage["lock"]:
        today = date.today().isoformat()
        if usage["day"] != today:
            usage["day"], usage["count"] = today, 0
        if usage["count"] >= DAILY_SEARCH_LIMIT:
            return False
        usage["count"] += 1
        return True


def _take_session(kind: str, per_hour: int) -> bool:
    now = time.time()
    stamps = [t for t in st.session_state.get(kind, []) if now - t < 3600]
    allowed = len(stamps) < per_hour
    if allowed:
        stamps.append(now)
    st.session_state[kind] = stamps
    return allowed


# ── Text helpers ──────────────────────────────────────────────────────────────

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|$<>~])")


def md(text: str) -> str:
    """Escape text from OpenAlex or the LLM so it renders literally in markdown
    ($ would otherwise start LaTeX, [ ] a link, etc.)."""
    return _MD_SPECIAL.sub(r"\\\1", text or "")


def safe_url(url: str) -> str:
    return url.replace(" ", "%20").replace("(", "%28").replace(")", "%29")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(question: str, model_name: str, top_k: int) -> dict | None:
    with st.status("Working on your question...", expanded=True) as status:
        st.write("Breaking the question into aspects...")
        try:
            plan = cached_plan(question)
        except LLMError:
            plan = {"aspects": [], "sub_queries": [question]}
            st.write("⚠️ Couldn't analyze the question; searching it as-is without coverage analysis.")

        st.write(f"Searching OpenAlex with {len(plan['sub_queries'])} queries and reranking...")
        try:
            papers = cached_rerank(question, tuple(plan["sub_queries"]), model_name)
        except NoResults:
            status.update(label="No papers found", state="error")
            st.error("No papers with abstracts were found (or OpenAlex is unavailable). Try rephrasing.")
            return None

        assessments, gaps = None, None
        if plan["aspects"]:
            st.write(f"Checking how well the top {top_k} papers cover each aspect...")
            try:
                assessments = cached_assess(question, plan["aspects"], papers[:top_k])
                st.write("Looking for possible gaps...")
                gaps = cached_gaps(plan["aspects"], assessments)
            except LLMError:
                st.write("⚠️ Coverage analysis is unavailable right now; showing ranked papers only.")

        status.update(label=f"Found {len(papers)} candidate papers", state="complete", expanded=False)

    if assessments:
        for paper, analysis in zip(papers, assessments):
            paper["analysis"] = analysis
    return {
        "question": question,
        "model": model_name,
        "plan": plan,
        "papers": papers,
        "gaps": gaps,
    }


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_gaps(gaps: dict, aspects: list[dict]) -> None:
    with st.expander("🔍 Possible gaps", expanded=True):
        st.caption(
            "Leads to verify, not proof: the topic may be studied under other wording, "
            "in full texts, or in papers OpenAlex doesn't index."
        )
        shown = False
        if gaps.get("full_count") is not None:
            shown = True
            terms = ", ".join(f"*{md(a['keywords'])}*" for a in aspects)
            st.markdown(f"**{gaps['full_count']:,}** papers mention all of {terms} in their title or abstract.")
            for b in gaps["bottlenecks"]:
                ratio = b["count_without"] / max(gaps["full_count"], 1)
                st.markdown(f"- Without **{md(b['aspect'])}**: {b['count_without']:,} papers ({ratio:,.0f}× more)")
            if gaps["full_count"] < 20:
                st.info("Few papers combine all of these aspects — this combination may be understudied.")
        if gaps.get("uncovered"):
            shown = True
            names = ", ".join(f"**{md(n)}**" for n in gaps["uncovered"])
            st.markdown(f"None of the top papers fully covers: {names}")
        if not shown:
            st.write("No obvious gaps: the top papers cover every aspect.")


def render_paper(paper: dict, question: str, n_queries: int) -> None:
    pid = paper["paperId"]
    with st.container(border=True):
        title = md(paper["title"])
        url = paper.get("url")
        st.markdown(f"**#{paper['ce_rank']}** · " + (f"[{title}]({safe_url(url)})" if url else title))

        authors = [a["name"] for a in paper.get("authors", [])]
        author_text = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
        meta = [
            str(paper.get("year") or "n.d."),
            f"{paper.get('citationCount') or 0:,} citations",
            author_text,
            f"found by {len(paper.get('matched_queries', []))}/{n_queries} searches",
            f"match score {paper['ce_score']:.2f}",
        ]
        st.caption(md(" · ".join(m for m in meta if m)))

        analysis = paper.get("analysis")
        if analysis and analysis.get("coverage") is not None:
            st.progress(analysis["coverage"] / 100, text=f"Covers {analysis['coverage']}% of your question")
            for a in analysis["aspects"]:
                line = f"{STATUS_ICON[a['status']]} **{md(a['name'])}**"
                if a["evidence"]:
                    line += f' — *"{md(a["evidence"])}"*'
                st.markdown(line)
            if analysis.get("note"):
                st.caption(f"Not addressed: {md(analysis['note'])}")

        with st.expander("Abstract"):
            st.write(md(paper["abstract"]))

        summarized = st.session_state.setdefault("summarized", set())
        if pid in summarized:
            try:
                st.info(md(cached_summary(question, pid, paper["title"], paper["abstract"])))
            except LLMError as e:
                summarized.discard(pid)
                st.error(f"Summary failed: {e}")
        elif st.button("Summarize", key=f"sum_{pid}"):
            if _take_session("summary_times", SESSION_SUMMARIES_PER_HOUR):
                summarized.add(pid)
                st.rerun()
            else:
                st.warning("Summary limit reached for this hour. Please try again later.")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📄 PaperRank")
    st.caption("Find papers that answer your research question — and spot what's missing.")
    st.divider()
    model_name = st.selectbox(
        "Reranker model",
        AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(DEFAULT_MODEL),
        format_func=lambda m: MODEL_LABELS.get(m, m),
    )
    top_k = st.slider("Papers to analyze in depth", 5, 20, 10,
                      help="Top papers checked for coverage of each aspect. More is slower.")


# ── Main ──────────────────────────────────────────────────────────────────────

st.title("PaperRank")
with st.form("search"):
    question = st.text_area(
        "Your research question",
        placeholder="e.g. effects of microplastics on gut microbiome in zebrafish larvae",
        max_chars=MAX_QUESTION_CHARS,
        height=90,
    )
    submitted = st.form_submit_button("Search", type="primary")

if submitted:
    question = question.strip()
    if len(question) < 5:
        st.warning("Please enter a longer question.")
    elif not _take_session("search_times", SESSION_SEARCHES_PER_HOUR):
        st.warning("You've reached the search limit for this hour. Please try again later.")
    elif not _take_daily_search():
        st.warning("PaperRank has reached its daily search limit. Please come back tomorrow.")
    else:
        result = run_pipeline(question, model_name, top_k)
        if result:
            st.session_state["result"] = result
            st.session_state.pop("summarized", None)

result = st.session_state.get("result")
if not result:
    st.info(
        "Ask a question in plain language. PaperRank breaks it into aspects, searches OpenAlex "
        "several ways, reranks the results, and shows which papers cover which parts of your "
        "question — with quotes as evidence."
    )
    st.stop()

plan, papers = result["plan"], result["papers"]

st.subheader(f'Results for: "{md(result["question"])}"')
if plan["aspects"]:
    st.markdown("**Aspects:** " + " · ".join(f"`{a['name']}`" for a in plan["aspects"]))
with st.expander("Searches run"):
    for q in plan["sub_queries"]:
        st.markdown(f"- {md(q)}")
    st.caption(f"Reranked with {MODEL_LABELS.get(result['model'], result['model'])}")

if result["gaps"]:
    render_gaps(result["gaps"], plan["aspects"])

# Display-only controls: never re-run search or analysis.
col_sort, col_year = st.columns(2)
analyzed = [p for p in papers if p.get("analysis")]
with col_sort:
    sort_by = st.radio(
        "Sort analyzed papers by", ["Match score", "Coverage"],
        horizontal=True, disabled=not analyzed,
    )
years = sorted({p["year"] for p in papers if p.get("year")})
year_lo, year_hi = (years[0], years[-1]) if years else (None, None)
with col_year:
    if years and year_lo < year_hi:
        year_lo, year_hi = st.slider("Publication years", year_lo, year_hi, (year_lo, year_hi))


def in_years(p: dict) -> bool:
    return year_lo is None or p.get("year") is None or year_lo <= p["year"] <= year_hi


analyzed = [p for p in analyzed if in_years(p)]
rest = [p for p in papers if not p.get("analysis") and in_years(p)]
if sort_by == "Coverage":
    analyzed.sort(key=lambda p: (-p["analysis"]["coverage"], p["ce_rank"]))

n_queries = len(plan["sub_queries"])
top = analyzed if analyzed else rest[:top_k]
rest = rest if analyzed else rest[top_k:]

if not top and not rest:
    st.warning("No papers in the selected year range.")
for paper in top:
    render_paper(paper, result["question"], n_queries)

if rest:
    with st.expander(f"More candidates ({len(rest)}, not analyzed)"):
        for paper in rest:
            render_paper(paper, result["question"], n_queries)

with st.expander("How ranking works"):
    st.markdown(
        "1. An LLM breaks your question into **aspects** and writes several **search queries**.\n"
        "2. Each query runs as a semantic search on **OpenAlex** (200M+ papers); results are merged.\n"
        "3. A **CrossEncoder** reads your question next to every paper and reranks them (*match score*).\n"
        "4. The LLM checks each top paper against each aspect and must **quote the abstract** as "
        "evidence; quotes not found in the abstract are discarded. *Coverage* = covered (1) + "
        "partial (½), averaged over aspects.\n"
        "5. **Possible gaps** come from keyword counts on OpenAlex: how many papers mention all "
        "aspects together, and how many when each aspect is dropped.\n\n"
        "Based on abstracts only. Always verify against the full paper."
    )
