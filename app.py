"""PaperRank web app (Gradio). Run locally with:  python app.py"""
import copy
import html
import json
import os
import threading
import time
from datetime import date
from functools import partial

from dotenv import load_dotenv
load_dotenv()

import gradio as gr

from analyzer import LLMError, assess_coverage, decompose, find_gaps, summarize_paper
from reranker import AVAILABLE_MODELS, DEFAULT_MODEL, load_model, rerank
from retriever import multi_search


MAX_QUESTION_CHARS = 500
CACHE_TTL = 24 * 3600
CACHE_MAX_ENTRIES = 256
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


class NoResults(Exception):
    pass


# ── Caching ───────────────────────────────────────────────────────────────────
# Same idea as Streamlit's st.cache_data: identical inputs within 24h reuse the
# result, exceptions are never cached, and callers get a copy they can modify.

def ttl_cache(fn):
    store: dict[str, tuple[float, object]] = {}
    lock = threading.Lock()

    def wrapper(*args):
        key = json.dumps(args, sort_keys=True, default=str)
        with lock:
            hit = store.get(key)
        if hit and time.time() - hit[0] < CACHE_TTL:
            return copy.deepcopy(hit[1])
        value = fn(*args)
        with lock:
            if len(store) >= CACHE_MAX_ENTRIES:
                store.pop(min(store, key=lambda k: store[k][0]))  # drop the oldest
            store[key] = (time.time(), value)
        return copy.deepcopy(value)

    return wrapper


_models: dict = {}
_models_lock = threading.Lock()


def get_model(name: str):
    # Lock so two visitors picking the same model don't load it twice.
    with _models_lock:
        if name not in _models:
            _models[name] = load_model(name)
        return _models[name]


@ttl_cache
def cached_plan(question: str) -> dict:
    plan = decompose(question)
    if not plan["aspects"]:
        raise LLMError("Could not break the question into aspects.")
    return plan


@ttl_cache
def cached_search(sub_queries: list[str]) -> list[dict]:
    papers = multi_search(sub_queries)
    if not papers:
        raise NoResults()
    return papers


@ttl_cache
def cached_rerank(question: str, sub_queries: list[str], model_name: str) -> list[dict]:
    return rerank(question, cached_search(sub_queries), get_model(model_name))


@ttl_cache
def cached_assess(question: str, aspects: list[dict], papers: list[dict]) -> list[dict]:
    return assess_coverage(question, aspects, papers)


@ttl_cache
def cached_gaps(aspects: list[dict], assessments: list[dict]) -> dict:
    return find_gaps(aspects, assessments)


@ttl_cache
def cached_summary(question: str, paper_id: str, title: str, abstract: str) -> str:
    return summarize_paper(question, {"title": title, "abstract": abstract})


# ── Rate limiting ─────────────────────────────────────────────────────────────

_daily = {"lock": threading.Lock(), "day": "", "count": 0}  # shared by all visitors


def take_daily_search() -> bool:
    with _daily["lock"]:
        today = date.today().isoformat()
        if _daily["day"] != today:
            _daily["day"], _daily["count"] = today, 0
        if _daily["count"] >= DAILY_SEARCH_LIMIT:
            return False
        _daily["count"] += 1
        return True


def take_session(limits: dict, kind: str, per_hour: int) -> tuple[bool, dict]:
    """Per-browser-session sliding window. Returns (allowed, updated limits)."""
    now = time.time()
    stamps = [t for t in limits.get(kind, []) if now - t < 3600]
    allowed = len(stamps) < per_hour
    if allowed:
        stamps.append(now)
    return allowed, {**limits, kind: stamps}


# ── HTML rendering ────────────────────────────────────────────────────────────
# Everything from OpenAlex or the LLM goes through esc(), so titles or abstracts
# containing <, & or quotes show up as text and can't inject markup.

def esc(text) -> str:
    return html.escape(str(text or ""), quote=True)


def link(url: str, text: str) -> str:
    if url and url.startswith(("https://", "http://")):
        return f'<a href="{esc(url)}" target="_blank" rel="noopener noreferrer">{esc(text)}</a>'
    return esc(text)


def paper_card(paper: dict, n_queries: int, summary: str | None) -> str:
    authors = [a["name"] for a in paper.get("authors", [])]
    author_text = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
    meta = [
        str(paper.get("year") or "n.d."),
        f"{paper.get('citationCount') or 0:,} citations",
        author_text,
        f"found by {len(paper.get('matched_queries', []))}/{n_queries} searches",
        f"match score {paper['ce_score']:.2f}",
    ]
    parts = [
        '<div class="pr-card">',
        f'<div class="pr-title"><span class="pr-rank">#{paper["ce_rank"]}</span> '
        f'{link(paper.get("url"), paper["title"])}</div>',
        f'<div class="pr-meta">{esc(" · ".join(m for m in meta if m))}</div>',
    ]

    analysis = paper.get("analysis")
    if analysis and analysis.get("coverage") is not None:
        cov = analysis["coverage"]
        parts.append(
            f'<div class="pr-bar"><div style="width:{cov}%"></div></div>'
            f'<div class="pr-cov">Covers {cov}% of your question</div><ul class="pr-aspects">'
        )
        for a in analysis["aspects"]:
            line = f'{STATUS_ICON[a["status"]]} <b>{esc(a["name"])}</b>'
            if a["evidence"]:
                line += f' — <i>"{esc(a["evidence"])}"</i>'
            parts.append(f"<li>{line}</li>")
        parts.append("</ul>")
        if analysis.get("note"):
            parts.append(f'<div class="pr-note">Not addressed: {esc(analysis["note"])}</div>')

    parts.append(f'<details><summary>Abstract</summary><p>{esc(paper["abstract"])}</p></details>')
    if summary:
        parts.append(f'<div class="pr-summary"><b>Summary:</b> {esc(summary)}</div>')
    parts.append("</div>")
    return "".join(parts)


def overview_html(result: dict, notes: list[str]) -> str:
    plan, gaps = result["plan"], result["gaps"]
    parts = [f'<h3 class="pr-h">Results for: "{esc(result["question"])}"</h3>']
    for note in notes:
        parts.append(f'<div class="pr-warn">⚠️ {esc(note)}</div>')
    if plan["aspects"]:
        chips = "".join(f'<span class="pr-chip">{esc(a["name"])}</span>' for a in plan["aspects"])
        parts.append(f"<div><b>Aspects:</b> {chips}</div>")
    searches = "".join(f"<li>{esc(q)}</li>" for q in plan["sub_queries"])
    parts.append(
        f"<details><summary>Searches run ({len(plan['sub_queries'])})</summary><ul>{searches}</ul>"
        f'<div class="pr-meta">Reranked with {esc(MODEL_LABELS.get(result["model"], result["model"]))}'
        f" · {len(result['papers'])} candidate papers</div></details>"
    )
    if gaps:
        parts.append(gaps_html(gaps, plan["aspects"]))
    return "".join(parts)


def gaps_html(gaps: dict, aspects: list[dict]) -> str:
    parts = [
        '<div class="pr-gaps"><b>🔍 Possible gaps</b>'
        '<div class="pr-meta">Leads to verify, not proof: the topic may be studied under other '
        "wording, in full texts, or in papers OpenAlex doesn't index.</div>"
    ]
    shown = False
    if gaps.get("full_count") is not None:
        shown = True
        terms = ", ".join(f"<i>{esc(a['keywords'])}</i>" for a in aspects)
        parts.append(f"<p><b>{gaps['full_count']:,}</b> papers mention all of {terms} in their title or abstract.</p><ul>")
        for b in gaps["bottlenecks"]:
            ratio = b["count_without"] / max(gaps["full_count"], 1)
            parts.append(f"<li>Without <b>{esc(b['aspect'])}</b>: {b['count_without']:,} papers ({ratio:,.0f}× more)</li>")
        parts.append("</ul>")
        if gaps["full_count"] < 20:
            parts.append("<p>Few papers combine all of these aspects — this combination may be understudied.</p>")
    if gaps.get("uncovered"):
        shown = True
        names = ", ".join(f"<b>{esc(n)}</b>" for n in gaps["uncovered"])
        parts.append(f"<p>None of the top papers fully covers: {names}</p>")
    if not shown:
        parts.append("<p>No obvious gaps: the top papers cover every aspect.</p>")
    parts.append("</div>")
    return "".join(parts)


# ── Event handlers ────────────────────────────────────────────────────────────

def run_search(question, model_name, top_k, limits, progress=gr.Progress()):
    """Search button: run the whole pipeline and store the result in session state."""
    question = (question or "").strip()
    top_k = int(top_k)
    if len(question) < 5:
        gr.Warning("Please enter a longer question.")
        return {limits_state: limits}
    allowed, limits = take_session(limits, "search", SESSION_SEARCHES_PER_HOUR)
    if not allowed:
        gr.Warning("You've reached the search limit for this hour. Please try again later.")
        return {limits_state: limits}
    if not take_daily_search():
        gr.Warning("PaperRank has reached its daily search limit. Please come back tomorrow.")
        return {limits_state: limits}

    notes = []
    progress(0.05, desc="Breaking the question into aspects...")
    try:
        plan = cached_plan(question)
    except LLMError:
        plan = {"aspects": [], "sub_queries": [question]}
        notes.append("Couldn't analyze the question; searched it as-is without coverage analysis.")

    progress(0.25, desc=f"Searching OpenAlex with {len(plan['sub_queries'])} queries and reranking...")
    try:
        papers = cached_rerank(question, plan["sub_queries"], model_name)
    except NoResults:
        gr.Warning("No papers with abstracts were found (or OpenAlex is unavailable). Try rephrasing.")
        return {limits_state: limits}

    gaps = None
    if plan["aspects"]:
        progress(0.65, desc=f"Checking how well the top {top_k} papers cover each aspect...")
        try:
            assessments = cached_assess(question, plan["aspects"], papers[:top_k])
            progress(0.9, desc="Looking for possible gaps...")
            gaps = cached_gaps(plan["aspects"], assessments)
            for paper, analysis in zip(papers, assessments):
                paper["analysis"] = analysis
        except LLMError:
            notes.append("Coverage analysis is unavailable right now; showing ranked papers only.")

    result = {"question": question, "model": model_name, "plan": plan,
              "papers": papers, "gaps": gaps, "top_k": top_k}
    years = sorted({p["year"] for p in papers if p.get("year")})
    has_range = len(years) > 1
    lo, hi = (years[0], years[-1]) if years else (0, 1)
    analyzed = any(p.get("analysis") for p in papers)
    return {
        result_state: result,
        summaries_state: {},
        limits_state: limits,
        overview: gr.update(value=overview_html(result, notes)),
        controls: gr.update(visible=True),
        sort_by: gr.update(value="Match score", interactive=analyzed),
        year_from: gr.update(minimum=lo, maximum=hi, value=lo, visible=has_range),
        year_to: gr.update(minimum=lo, maximum=hi, value=hi, visible=has_range),
    }


def run_summary(paper_id, result, summaries, limits):
    """Summarize one paper (button on a card, or the picker for other candidates)."""
    if not result or not paper_id:
        return summaries, limits
    paper = next((p for p in result["papers"] if p["paperId"] == paper_id), None)
    if paper is None or paper_id in summaries:
        return summaries, limits
    allowed, limits = take_session(limits, "summary", SESSION_SUMMARIES_PER_HOUR)
    if not allowed:
        gr.Warning("Summary limit reached for this hour. Please try again later.")
        return summaries, limits
    try:
        text = cached_summary(result["question"], paper_id, paper["title"], paper["abstract"])
    except LLMError as e:
        gr.Warning(f"Summary failed: {e}")
        return summaries, limits
    return {**summaries, paper_id: text}, limits


# ── Layout ────────────────────────────────────────────────────────────────────

CSS = """
.pr-card { padding: 4px 2px; line-height: 1.5; }
.pr-title { font-weight: 600; font-size: 1.05em; }
.pr-rank { color: var(--body-text-color-subdued); margin-right: 4px; }
.pr-meta { color: var(--body-text-color-subdued); font-size: 0.88em; margin: 2px 0 6px; }
.pr-bar { height: 8px; border-radius: 4px; background: var(--background-fill-secondary);
          border: 1px solid var(--border-color-primary); overflow: hidden; margin-top: 6px; }
.pr-bar > div { height: 100%; background: var(--color-accent); }
.pr-cov { font-size: 0.88em; margin: 2px 0 4px; }
.pr-aspects { margin: 4px 0; padding-left: 0; list-style: none; }
.pr-aspects li { margin: 2px 0; }
.pr-note { color: var(--body-text-color-subdued); font-size: 0.88em; }
.pr-summary { margin-top: 8px; padding: 8px 10px; border-radius: 6px;
              background: var(--background-fill-secondary); }
.pr-chip { display: inline-block; padding: 1px 8px; margin: 2px 4px 2px 0; border-radius: 10px;
           border: 1px solid var(--border-color-primary); font-size: 0.9em; }
.pr-gaps { margin-top: 10px; padding: 10px 12px; border-radius: 8px;
           border: 1px solid var(--border-color-primary); }
.pr-warn { margin: 4px 0; }
.pr-h { margin: 0 0 6px; }
details summary { cursor: pointer; }
"""

INTRO = (
    "Ask a question in plain language. PaperRank breaks it into aspects, searches OpenAlex "
    "several ways, reranks the results, and shows which papers cover which parts of your "
    "question — with quotes as evidence."
)

HOW_IT_WORKS = """
1. An LLM breaks your question into **aspects** and writes several **search queries**.
2. Each query runs as a semantic search on **OpenAlex** (200M+ papers); results are merged.
3. A **CrossEncoder** reads your question next to every paper and reranks them (*match score*).
4. The LLM checks each top paper against each aspect and must **quote the abstract** as
   evidence; quotes not found in the abstract are discarded. *Coverage* = covered (1) +
   partial (½), averaged over aspects.
5. **Possible gaps** come from keyword counts on OpenAlex: how many papers mention all
   aspects together, and how many when each aspect is dropped.

Based on abstracts only. Always verify against the full paper.
"""

with gr.Blocks(title="PaperRank") as demo:
    result_state = gr.State(None)
    summaries_state = gr.State({})
    limits_state = gr.State({})

    with gr.Sidebar():
        gr.Markdown("## 📄 PaperRank\nFind papers that answer your research question — and spot what's missing.")
        model_dd = gr.Dropdown(
            choices=[(MODEL_LABELS.get(m, m), m) for m in AVAILABLE_MODELS],
            value=DEFAULT_MODEL, label="Reranker model",
        )
        top_k_slider = gr.Slider(5, 20, value=10, step=1, label="Papers to analyze in depth",
                                 info="Top papers checked for coverage of each aspect. More is slower.")

    gr.Markdown("# PaperRank")
    question_box = gr.Textbox(
        label="Your research question", lines=3, max_length=MAX_QUESTION_CHARS,
        placeholder="e.g. effects of microplastics on gut microbiome in zebrafish larvae",
    )
    search_btn = gr.Button("Search", variant="primary")

    overview = gr.HTML(f"<p>{esc(INTRO)}</p>")

    # Display-only controls: they never re-run search or analysis.
    with gr.Row(visible=False) as controls:
        sort_by = gr.Radio(["Match score", "Coverage"], value="Match score",
                           label="Sort analyzed papers by")
        year_from = gr.Slider(0, 1, step=1, label="From year", visible=False)
        year_to = gr.Slider(0, 1, step=1, label="To year", visible=False)

    @gr.render(
        inputs=[result_state, summaries_state, sort_by, year_from, year_to],
        triggers=[result_state.change, summaries_state.change, sort_by.change,
                  year_from.release, year_to.release],
    )
    def show_papers(result, summaries, sort_mode, y_from, y_to):
        if not result:
            return
        papers = result["papers"]
        lo, hi = sorted((int(y_from or 0), int(y_to or 9999)))
        years = [p["year"] for p in papers if p.get("year")]
        if not years or min(years) == max(years):
            lo, hi = 0, 9999  # sliders hidden: no filtering

        def in_years(p):
            return p.get("year") is None or lo <= p["year"] <= hi

        analyzed = [p for p in papers if p.get("analysis") and in_years(p)]
        rest = [p for p in papers if not p.get("analysis") and in_years(p)]
        if sort_mode == "Coverage":
            analyzed.sort(key=lambda p: (-p["analysis"]["coverage"], p["ce_rank"]))
        top_k = result["top_k"]
        top, rest = (analyzed, rest) if analyzed else (rest[:top_k], rest[top_k:])
        n_queries = len(result["plan"]["sub_queries"])

        if not top and not rest:
            gr.Markdown("No papers in the selected year range.")
        for paper in top:
            pid = paper["paperId"]
            with gr.Group():
                gr.HTML(paper_card(paper, n_queries, summaries.get(pid)), padding=True)
                if pid not in summaries:
                    btn = gr.Button("Summarize", size="sm", key=f"sum-{pid}")
                    btn.click(partial(run_summary, pid),
                              inputs=[result_state, summaries_state, limits_state],
                              outputs=[summaries_state, limits_state])

        if rest:
            with gr.Accordion(f"More candidates ({len(rest)}, not analyzed)", open=False,
                              key="more", preserved_by_key=["open"]):
                with gr.Row():
                    pick = gr.Dropdown(
                        choices=[(f"#{p['ce_rank']} {p['title'][:90]}", p["paperId"])
                                 for p in rest if p["paperId"] not in summaries],
                        label="Summarize one of these", scale=4, key="pick",
                    )
                    pick_btn = gr.Button("Summarize", size="sm", scale=1)
                pick_btn.click(run_summary, inputs=[pick, result_state, summaries_state, limits_state],
                               outputs=[summaries_state, limits_state])
                gr.HTML("<hr>".join(paper_card(p, n_queries, summaries.get(p["paperId"])) for p in rest),
                        padding=True)

    with gr.Accordion("How ranking works", open=False):
        gr.Markdown(HOW_IT_WORKS)

    search_btn.click(
        run_search,
        inputs=[question_box, model_dd, top_k_slider, limits_state],
        outputs=[result_state, summaries_state, limits_state, overview, controls,
                 sort_by, year_from, year_to],
        concurrency_limit=2,  # reranking is CPU-heavy; queue the rest
    )


if __name__ == "__main__":
    demo.queue(max_size=20).launch(theme=gr.themes.Soft(), css=CSS)
