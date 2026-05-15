import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from retriever import fetch_papers
from reranker import load_model, rerank, AVAILABLE_MODELS, DEFAULT_MODEL

st.set_page_config(page_title="PaperRank", layout="wide")

TIER_CONFIG = [
    ("Perfectly Relevant", "#28a745", 0.90),
    ("Relevant",           "#e6a817", 0.70),
    ("Somewhat Relevant",  "#fd7e14", 0.50),
    ("Not Relevant",       "#6c757d", float("-inf")),
]


@st.cache_resource
def get_model(model_name: str):
    return load_model(model_name)


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
    st.caption("CrossEncoder reranking on live academic papers.")
    st.divider()

    query = st.text_input("Search query", placeholder="e.g. transformer models for information retrieval")
    limit = st.slider("Candidates to fetch", min_value=10, max_value=200, value=50, step=10)
    k     = st.slider("Top-k results",       min_value=5,  max_value=20,  value=10)

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
    with st.spinner(f"[1/3] Fetching up to {limit} papers from OpenAlex..."):
        try:
            baseline_papers = fetch_papers(query, limit=limit)
        except Exception as e:
            st.error(f"Fetch failed: {e}")
            st.stop()

    st.success(f"[1/3] Retrieved {len(baseline_papers)} papers with abstracts.")
    st.session_state["baseline_papers"] = baseline_papers
    st.session_state["active_query"]    = query

    with st.spinner(f"[2/3] Reranking {len(baseline_papers)} papers with {model_name}..."):
        model = get_model(model_name)
        st.session_state["reranked_papers"] = rerank(query, baseline_papers, model)
        st.session_state["active_model"]    = model_name

    # Rerun so the year slider updates to the actual bounds before results render
    st.rerun()


# ── Pipeline: re-rerank when model changes (no new fetch needed) ───────────────
if (
    not run
    and "baseline_papers" in st.session_state
    and st.session_state.get("active_model") != model_name
):
    with st.spinner(f"Switching model to {model_name}..."):
        model = get_model(model_name)
        st.session_state["reranked_papers"] = rerank(
            st.session_state["active_query"],
            st.session_state["baseline_papers"],
            model,
        )
        st.session_state["active_model"] = model_name
    st.rerun()


# ── Main area ──────────────────────────────────────────────────────────────────
st.title("PaperRank")
st.caption("CrossEncoder reranking on live academic papers.")

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

# Tier counts from the full year-filtered reranked list (not just top k)
tier_counts = {label: 0 for label, _, _ in TIER_CONFIG}
for paper in filtered_reranked:
    score = paper.get("ce_score", 0.0)
    for label, _, threshold in TIER_CONFIG:
        if score >= threshold:
            tier_counts[label] += 1
            break

# Rank-change lookups (CE rank vs original OpenAlex rank)
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


# ── Card renderer ──────────────────────────────────────────────────────────────
def render_ce_card(paper: dict) -> None:
    rank  = paper["ce_rank"]
    pid   = paper.get("paperId", "")
    title = (paper.get("title") or "")[:80]
    url   = paper.get("url", "")
    title_md = f"[{title}]({url})" if url else title
    score = paper.get("ce_score", 0.0)

    oa_r = oa_rank_by_id.get(pid)
    note = (f" {'↑' if rank < oa_r else '↓'} from OA #{oa_r}") if oa_r and oa_r != rank else ""

    st.markdown(
        f"**#{rank}**{note} {tier_badge(score)}  \n{title_md}  \n"
        f"`{paper.get('year', 'N/A')}` · {paper.get('citationCount', 'N/A')} citations · score: `{score}`",
        unsafe_allow_html=True,
    )


# ── Top-K results ──────────────────────────────────────────────────────────────
top_reranked = filtered_reranked[:k]

if not top_reranked:
    st.warning("No papers match the selected year range. Adjust the slider in the sidebar.")
    st.stop()

for paper in top_reranked:
    render_ce_card(paper)

# ── Expander: remaining papers (rank k+1 to end of filtered list) ──────────────
rest_reranked = filtered_reranked[k:]

if rest_reranked:
    total = len(filtered_reranked)
    with st.expander(f"Show all {total} papers (ranks {k + 1}–{total})"):
        for paper in rest_reranked:
            render_ce_card(paper)
