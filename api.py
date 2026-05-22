import os

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from retriever import fetch_papers
from reranker import load_model, rerank, AVAILABLE_MODELS, DEFAULT_MODEL

app = FastAPI(title="PaperRank API")

_model_cache: dict = {}


def _get_model(model_name: str):
    if model_name not in _model_cache:
        _model_cache[model_name] = load_model(model_name)
    return _model_cache[model_name]


class SearchRequest(BaseModel):
    query: str
    limit: int = 50
    model_name: str = DEFAULT_MODEL


class SummarizeRequest(BaseModel):
    title: str
    abstract: str


class ResearchSummaryRequest(BaseModel):
    query: str
    papers: list[dict]


@app.post("/search")
def search(req: SearchRequest):
    if req.model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model. Choose from: {AVAILABLE_MODELS}",
        )
    try:
        papers = fetch_papers(req.query, limit=req.limit)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAlex fetch failed: {e}")
    model = _get_model(req.model_name)
    reranked = rerank(req.query, papers, model)
    return {"papers": reranked}


@app.post("/summarize")
def summarize(req: SummarizeRequest):
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
    prompt = (
        "Summarize this academic paper in 2-3 sentences focusing "
        "on the key contribution and findings:\n"
        f"Title: {req.title}\n"
        f"Abstract: {req.abstract}"
    )
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
    )
    return {"summary": resp.choices[0].message.content}


@app.post("/research-summary")
def research_summary(req: ResearchSummaryRequest):
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
    papers_text = "\n".join(
        f"{i}. {p.get('title', 'N/A')}: {p.get('abstract') or 'No abstract available'}"
        for i, p in enumerate(req.papers, 1)
    )
    prompt = (
        f"Based on these top 5 papers retrieved for the query: {req.query}\n\n"
        f"Papers:\n{papers_text}\n\n"
        "Write a concise 150 word synthesis of the key themes and "
        "findings relevant to the query. Focus on what the papers "
        "collectively say about the topic."
    )
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )
    return {"summary": resp.choices[0].message.content}

@app.get("/health")
def health_check():
    return {"status": "ok"}