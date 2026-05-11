import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from retriever import retrieve
from reranker import load_model, rerank
from evaluate import evaluate

st.set_page_config(page_title="PaperRank", layout="wide")

MODEL_NAME = "BAAI/bge-reranker-v2-m3"


@st.cache_resource
def get_model():
    return load_model(MODEL_NAME)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("PaperRank")
    st.caption("BM25 vs CrossEncoder reranking on live papers from OpenAlex.")
    st.divider()
    query = st.text_input("Search query", placeholder="e.g. transformer models for information retrieval")
    limit = st.slider("Candidates to fetch", min_value=10, max_value=200, value=50, step=10)
    k = st.slider("Top-k results", min_value=5, max_value=20, value=10)
    run = st.button("Run pipeline", type="primary", use_container_width=True, disabled=not query.strip())

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("PaperRank")
st.caption("Compare BM25 baseline retrieval with CrossEncoder reranking on live OpenAlex papers.")

if not query.strip():
    st.info("Enter a query in the sidebar to get started.")
    st.stop()

if not run:
    st.stop()

# Stage 1 — fetch and BM25-rank
with st.spinner(f"[1/3] Fetching up to {limit} papers from OpenAlex..."):
    try:
        bm25_papers = retrieve(query, limit=limit)
    except Exception as e:
        st.error(f"Fetch failed: {e}")
        st.stop()

st.success(f"[1/3] Retrieved {len(bm25_papers)} papers with abstracts.")

# Stage 2 — load model (cached) and rerank
with st.spinner("[2/3] Loading CrossEncoder model and scoring pairs..."):
    model = get_model()
    reranked_papers = rerank(query, bm25_papers, model)

st.success(f"[2/3] Scored {len(reranked_papers)} (query, paper) pairs.")

# Stage 3 — side-by-side results
st.subheader(f"[3/3] Top-{k} results for: \"{query}\"")

top_bm25 = bm25_papers[:k]
top_ce = reranked_papers[:k]

# Build rank-change lookups so each column can flag moved papers
ce_rank_by_id = {p["paperId"]: p["ce_rank"] for p in top_ce if p.get("paperId")}
bm25_rank_by_id = {p["paperId"]: p["bm25_rank"] for p in top_bm25 if p.get("paperId")}

col_bm25, col_ce = st.columns(2)

with col_bm25:
    st.markdown("#### BM25 Ranking")
    for paper in top_bm25:
        rank = paper["bm25_rank"]
        pid = paper.get("paperId", "")
        title = (paper.get("title") or "")[:80]
        score = paper.get("bm25_score", "")

        ce_rank = ce_rank_by_id.get(pid)
        if ce_rank is not None and ce_rank != rank:
            direction = "up" if ce_rank < rank else "down"
            note = f" — moved {direction} to CE #{ce_rank}"
        else:
            note = ""

        url = paper.get("url", "")
        title_md = f"[{title}]({url})" if url else title
        st.markdown(f"**#{rank}**{note}  \n{title_md}  \n`BM25: {score}`")

with col_ce:
    st.markdown("#### CrossEncoder Ranking")
    for paper in top_ce:
        rank = paper["ce_rank"]
        pid = paper.get("paperId", "")
        title = (paper.get("title") or "")[:80]
        score = paper.get("ce_score", "")

        bm25_rank = bm25_rank_by_id.get(pid)
        if bm25_rank is not None and bm25_rank != rank:
            direction = "up" if rank < bm25_rank else "down"
            note = f" — {direction} from BM25 #{bm25_rank}"
        else:
            note = ""

        url = paper.get("url", "")
        title_md = f"[{title}]({url})" if url else title
        st.markdown(f"**#{rank}**{note}  \n{title_md}  \n`CE: {score}`")

# ── Metrics cards ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("Evaluation Metrics")

metrics = evaluate(bm25_papers, reranked_papers, k=k)

m1, m2, m3, m4 = st.columns(4)
m1.metric(f"NDCG@{k} — BM25", f"{metrics['bm25_ndcg']:.4f}")
m2.metric(
    f"NDCG@{k} — CrossEncoder",
    f"{metrics['ce_ndcg']:.4f}",
    delta=round(metrics["ce_ndcg"] - metrics["bm25_ndcg"], 4),
)
m3.metric("MRR — BM25", f"{metrics['bm25_mrr']:.4f}")
m4.metric(
    "MRR — CrossEncoder",
    f"{metrics['ce_mrr']:.4f}",
    delta=round(metrics["ce_mrr"] - metrics["bm25_mrr"], 4),
)
