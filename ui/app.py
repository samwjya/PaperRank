import os

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

AVAILABLE_MODELS = [
    "BAAI/bge-reranker-v2-m3",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-electra-base",
]
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

st.set_page_config(page_title="PaperRank", layout="wide")

TIER_CONFIG = [
    ("Perfectly Relevant", "#28a745", 0.90),
    ("Relevant",           "#e6a817", 0.70),
    ("Somewhat Relevant",  "#fd7e14", 0.50),
    ("Not Relevant",       "#6c757d", float("-inf")),
]


# ── API helpers ────────────────────────────────────────────────────────────────

def _call_search(query: str, limit: int, model_name: str) -> list[dict]:
    resp = requests.post(
        f"{BACKEND_URL}/search",
        json={"query": query, "limit": limit, "model_name": model_name},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["papers"]


def _call_summarize(title: str, abstract: str) -> str:
    resp = requests.post(
        f"{BACKEND_URL}/summarize",
        json={"title": title, "abstract": abstract},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["summary"]


def _call_research_summary(query: str, papers: list[dict]) -> str:
    resp = requests.post(
        f"{BACKEND_URL}/research-summary",
        json={"query": query, "papers": papers},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["summary"]


def _api_error(e: Exception) -> str:
    if isinstance(e, requests.ConnectionError):
        return f"Cannot reach backend at {BACKEND_URL}. Is the API server running?"
    if isinstance(e, requests.HTTPError) and e.response is not None:
        detail = e.response.json().get("detail", str(e))
        return f"Backend error: {detail}"
    return f"Request failed: {e}"


# ── UI helpers ─────────────────────────────────────────────────────────────────

def tier_badge(score: float) -> str:
    if score >= 0.90:
        label, color = "Perfectly Relevant", "#28a745"
    elif score >= 0.70:
        label, color = "Relevant", "#e6a817"
    elif score >= 0.50:
        label, color = "Somewhat Relevant", "#fd7e14"
    else:
        label, color = "Not Relevant", "#6c757d"
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:12px;font-size:0.75em;font-weight:bold;">{label}</span>'
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("PaperRank")
    st.caption("Research Paper Finder")
    st.divider()

    query = st.text_input("Search query", placeholder="e.g. transformer models for information retrieval")
    limit = 50
    k = 10

    st.divider()
    model_name = st.selectbox(
        "Reranker model",
        AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(DEFAULT_MODEL),
    )

    if "reranked_papers" in st.session_state:
        _years = [
            p["year"] for p in st.session_state["reranked_papers"]
            if p.get("year") is not None
        ]
        year_min = min(_years) if _years else 2000
        year_max = max(_years) if _years else 2026
        year_range = st.slider(
            "Publication year range",
            min_value=year_min,
            max_value=year_max,
            value=(year_min, year_max),
        )

    run = st.button(
        "Run pipeline", type="primary",
        use_container_width=True,
        disabled=not query.strip(),
    )


# ── Pipeline: fetch + rerank on Run click ──────────────────────────────────────
if run and query.strip():
    st.session_state.pop("summary", None)
    st.session_state.pop("summary_query", None)
    with st.spinner(f"Fetching and reranking with {model_name}..."):
        try:
            papers = _call_search(query, limit, model_name)
        except Exception as e:
            st.error(_api_error(e))
            st.stop()

    st.session_state["reranked_papers"] = papers
    st.session_state["active_query"]    = query
    st.session_state["active_model"]    = model_name
    st.rerun()


# ── Re-search when model changes without a new Run click ───────────────────────
if (
    not run
    and "reranked_papers" in st.session_state
    and st.session_state.get("active_model") != model_name
):
    with st.spinner(f"Switching model to {model_name}..."):
        try:
            papers = _call_search(
                st.session_state["active_query"], limit, model_name
            )
        except Exception as e:
            st.error(_api_error(e))
            st.stop()
    st.session_state["reranked_papers"] = papers
    st.session_state["active_model"]    = model_name
    st.rerun()


# ── Main area ──────────────────────────────────────────────────────────────────
st.title("PaperRank")
st.caption("Research Paper Finder")
if "reranked_papers" not in st.session_state:
    st.info("Enter a query in the sidebar and click **Run pipeline** to get started.")
    st.stop()

reranked_papers = st.session_state["reranked_papers"]
active_model    = st.session_state["active_model"]
active_query    = st.session_state["active_query"]

# ── Year filter — display only, never re-fetches or re-reranks ─────────────────
year_lo, year_hi = year_range


def in_year_range(paper: dict) -> bool:
    yr = paper.get("year")
    return yr is None or year_lo <= yr <= year_hi


filtered_reranked = [p for p in reranked_papers if in_year_range(p)]

tier_counts = {label: 0 for label, _, _ in TIER_CONFIG}
for paper in filtered_reranked:
    score = paper.get("ce_score", 0.0)
    for label, _, threshold in TIER_CONFIG:
        if score >= threshold:
            tier_counts[label] += 1
            break

oa_rank_by_id = {p["paperId"]: p["openalex_rank"] for p in filtered_reranked if p.get("paperId")}

# ── Header ─────────────────────────────────────────────────────────────────────
st.subheader(f'Results for: "{active_query}"')
st.info(f"Active model: **{active_model}**")

badges_html = " &nbsp;|&nbsp; ".join(
    f'<span style="background:{color};color:white;padding:2px 8px;'
    f'border-radius:12px;font-size:0.8em;font-weight:bold;">'
    f'{tier_counts[label]} {label}</span>'
    for label, color, _ in TIER_CONFIG
    if tier_counts[label] > 0
)
st.markdown(badges_html, unsafe_allow_html=True)
st.caption(
    f"Counts from all {len(filtered_reranked)} year-filtered papers "
    f"({year_lo}–{year_hi}) · showing top {k}"
)

# ── Research Summary ───────────────────────────────────────────────────────────
summary_exists = (
    "summary" in st.session_state
    and st.session_state.get("summary_query") == active_query
)

if summary_exists:
    st.markdown(
        '<div style="background:#0d2137;border-left:4px solid #4a9eff;'
        'padding:16px 20px;border-radius:8px;margin:12px 0;">'
        '<div style="color:#4a9eff;font-weight:bold;margin-bottom:8px;">'
        'Research Summary</div>'
        f'<div style="color:#e0e0e0;line-height:1.6;">{st.session_state["summary"]}</div>'
        "</div>",
        unsafe_allow_html=True,
    )
else:
    if st.button("Generate Research Summary", type="secondary"):
        with st.spinner("Generating research summary..."):
            try:
                summary = _call_research_summary(active_query, filtered_reranked[:5])
            except Exception as e:
                st.error(_api_error(e))
                st.stop()
        st.session_state["summary"]       = summary
        st.session_state["summary_query"] = active_query
        st.rerun()


# ── Card renderer ──────────────────────────────────────────────────────────────
def render_ce_card(paper: dict) -> None:
    rank  = paper["ce_rank"]
    pid   = paper.get("paperId") or str(hash(paper.get("title", "") + str(rank)))
    title = (paper.get("title") or "")[:80]
    url   = paper.get("url", "")
    title_md = f"[{title}]({url})" if url else title
    score = paper.get("ce_score", 0.0)

    oa_r = oa_rank_by_id.get(paper.get("paperId", ""))
    if oa_r and oa_r != rank:
        diff = abs(oa_r - rank)
        arrow = '↑' if rank < oa_r else '↓'
        note = f"{title} (#{oa_r} → #{rank}){arrow}{diff}"
    else:
        note = ""

    st.markdown(
        f"**#{rank}**{ note} {tier_badge(score)}  \n{title_md}  \n"
        f"`{paper.get('year', 'N/A')}` · {paper.get('citationCount', 'N/A')} citations · score: `{score}`",
        unsafe_allow_html=True,
    )

    summary_key = f"paper_summary_{pid}"
    if summary_key in st.session_state:
        st.markdown(
            '<div style="background:#1a1a2e;border-left:3px solid #6c8ebf;'
            'padding:10px 14px;border-radius:6px;margin:6px 0 10px 0;">'
            f'<span style="color:#a8b8d0;font-size:0.9em;">{st.session_state[summary_key]}</span>'
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        if st.button("Summarize", key=f"summarize_{pid}", help="Generate a 2-3 sentence summary"):
            with st.spinner("Generating summary..."):
                try:
                    paper_summary = _call_summarize(
                        paper.get("title", "N/A"),
                        paper.get("abstract") or "No abstract available.",
                    )
                except Exception as e:
                    st.error(_api_error(e))
                    st.stop()
            st.session_state[summary_key] = paper_summary
            st.rerun()


# ── Top-K results ──────────────────────────────────────────────────────────────
top_reranked = filtered_reranked[:k]

if not top_reranked:
    st.warning("No papers match the selected year range. Adjust the slider in the sidebar.")
    st.stop()

for paper in top_reranked:
    render_ce_card(paper)

# ── Expander: remaining papers ──────────────────────────────────────────────────
rest_reranked = filtered_reranked[k:]

if rest_reranked:
    total = len(filtered_reranked)
    with st.expander(f"Show all {total} papers (ranks {k + 1}–{total})"):
        for paper in rest_reranked:
            render_ce_card(paper)
