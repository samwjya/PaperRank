import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from retriever import fetch_papers
from reranker import load_models, ensemble_rerank, ENSEMBLE_MODELS
from evaluate import evaluate

st.set_page_config(page_title="PaperRank", layout="wide")

@st.cache_resource
def get_models():
    return load_models()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Paper Reranking System")
    st.caption("OpenAlex baseline vs CrossEncoder ensemble reranking on live papers.")
    st.divider()
    query = st.text_input("Search query", placeholder="e.g. transformer models for information retrieval")
    limit = st.slider("Candidates to fetch", min_value=10, max_value=200, value=50, step=10)
    k = st.slider("Top-k results", min_value=5, max_value=20, value=10)
    run = st.button("Run pipeline", type="primary", use_container_width=True, disabled=not query.strip())

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("Paper Reranking System")
st.caption("Compare OpenAlex baseline retrieval with CrossEncoder ensemble reranking on live papers.")

if not query.strip():
    st.info("Enter a query in the sidebar to get started.")
    st.stop()

if not run:
    st.stop()

# Stage 1 — fetch papers from OpenAlex
with st.spinner(f"[1/3] Fetching papers from OpenAlex..."):
    try:
        baseline_papers = fetch_papers(query, limit=limit)
    except Exception as e:
        st.error(f"Fetch failed: {e}")
        st.stop()

st.success(f"[1/3] Retrieved {len(baseline_papers)} papers with abstracts.")

# Stage 2 — load models (cached) and rerank
with st.spinner(f"[2/3] Running ensemble reranking ({len(ENSEMBLE_MODELS)} models)..."):
    models = get_models()
    reranked_papers = ensemble_rerank(query, baseline_papers, models)
st.success(f"[2/3] Ensemble: scored {len(reranked_papers)} papers × {len(ENSEMBLE_MODELS)} models.")

# Stage 3 — side-by-side results
st.subheader(f"[3/3] Top-{k} results for: \"{query}\"")

top_baseline = baseline_papers[:k]
top_ensemble = reranked_papers[:k]

# Build rank-change lookups so each column can show movement
ensemble_rank_by_id = {p["paperId"]: p["ce_rank"] for p in top_ensemble if p.get("paperId")}
openalex_rank_by_id = {p["paperId"]: p["openalex_rank"] for p in top_baseline if p.get("paperId")}

col_baseline, col_ensemble = st.columns(2)

with col_baseline:
    st.markdown("#### OpenAlex Baseline")
    for paper in top_baseline:
        rank = paper["openalex_rank"]
        pid = paper.get("paperId", "")
        title = (paper.get("title") or "")[:80]

        ens_rank = ensemble_rank_by_id.get(pid)
        if ens_rank is not None and ens_rank != rank:
            direction = "up" if ens_rank < rank else "down"
            note = f" — moved {direction} to Ensemble #{ens_rank}"
        else:
            note = ""

        url = paper.get("url", "")
        title_md = f"[{title}]({url})" if url else title
        st.markdown(f"**#{rank}**{note}  \n{title_md}  \n`OpenAlex rank: {rank}`")

with col_ensemble:
    st.markdown("#### Ensemble Reranking")
    for paper in top_ensemble:
        rank = paper["ce_rank"]
        pid = paper.get("paperId", "")
        title = (paper.get("title") or "")[:80]

        oa_rank = openalex_rank_by_id.get(pid)
        if oa_rank is not None and oa_rank != rank:
            direction = "up" if rank < oa_rank else "down"
            note = f" — {direction} from OpenAlex #{oa_rank}"
        else:
            note = ""

        url = paper.get("url", "")
        title_md = f"[{title}]({url})" if url else title

        s1 = paper.get("ce_score_model1", "N/A")
        s2 = paper.get("ce_score_model2", "N/A")
        s3 = paper.get("ce_score_model3", "N/A")
        ens = paper.get("ce_score_ensemble", "N/A")
        score_md = f"`BGE: {s1}` `MiniLM: {s2}` `Electra: {s3}` **Ensemble: {ens}**"

        st.markdown(f"**#{rank}**{note}  \n{title_md}  \n{score_md}")

# ── Metrics cards ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("Evaluation Metrics")

metrics = evaluate(baseline_papers, reranked_papers, k=k)

m1, m2, m3, m4 = st.columns(4)
m1.metric(f"NDCG@{k} — OpenAlex", f"{metrics['openalex_ndcg']:.4f}")
m2.metric(
    f"NDCG@{k} — Ensemble",
    f"{metrics['ensemble_ndcg']:.4f}",
    delta=round(metrics["ensemble_ndcg"] - metrics["openalex_ndcg"], 4),
)
m3.metric("MRR — OpenAlex", f"{metrics['openalex_mrr']:.4f}")
m4.metric(
    "MRR — Ensemble",
    f"{metrics['ensemble_mrr']:.4f}",
    delta=round(metrics["ensemble_mrr"] - metrics["openalex_mrr"], 4),
)
