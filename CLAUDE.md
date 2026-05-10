# PaperRank
 
An academic paper reranking system built for the Forward Data Lab research task at UIUC.
Adapts the Multi-Reranker approach from the ACM-ICAIF '24 paper
"Multi-Reranker: Maximizing performance of retrieval-augmented generation in the FinanceRAG challenge"
and applies it to academic literature search using the Semantic Scholar API.
 
## Project Goal
 
Compare BM25 baseline retrieval vs. multi-model cross-encoder reranking on academic papers.
The system fetches live papers from Semantic Scholar, ranks them with BM25, then reranks
using cross-encoder models inspired by the FinanceRAG paper. Evaluate using NDCG@10 and MRR.
 
## Stack
 
- **Language**: Python 3.10+
- **Data Source**: Semantic Scholar API (free, no key required)
- **Baseline Retrieval**: rank_bm25
- **Reranking**: sentence-transformers CrossEncoder (local, no API key)
  - Primary model: BAAI/bge-reranker-v2-m3
- **Evaluation**: NDCG@10, MRR (manual relevance judgments)
- **CLI**: argparse
## Project Structure
 
```
paperrank/
├── CLAUDE.md           # This file
├── main.py             # CLI entry point
├── retriever.py        # Semantic Scholar API fetching + BM25 ranking
├── reranker.py         # CrossEncoder reranking logic (adapted from FinanceRAG)
├── evaluate.py         # NDCG@10 and MRR metrics + side-by-side comparison
├── requirements.txt    # Dependencies
└── README.md           # Project documentation
```
 
## Key Files
 
- `retriever.py` — fetches top 50 candidate papers from Semantic Scholar API, ranks with BM25
- `reranker.py` — scores each (query, title + abstract) pair with CrossEncoder, returns reranked list
- `evaluate.py` — computes NDCG@10 and MRR, prints side-by-side BM25 vs reranked comparison table
- `main.py` — CLI entry: takes a query string, runs full pipeline, pretty prints results
## Architecture
 
```
User Query
    │
    ▼
Semantic Scholar API  (top 50 candidates)
    │
    ▼
BM25 Baseline Ranking
    │
    ▼
CrossEncoder Reranker  (BAAI/bge-reranker-v2-m3, runs locally)
    │
    ▼
Side-by-side Comparison  (BM25 rank vs Reranked rank, with scores)
    │
    ▼
NDCG@10 / MRR Evaluation
```
 
## Implementation Notes
 
- All models run **locally** — no OpenAI, Anthropic, or any paid API calls
- Reranker input: concatenate paper title + abstract as the document text
- BM25 tokenization: lowercase + whitespace split on title + abstract
- Fetch 50 candidates from Semantic Scholar, rerank to top 10 for evaluation
- Always show both BM25 and reranked results side-by-side for demo clarity
- Use `pandas` DataFrame for pretty printing comparison tables
## Reference Paper
 
- **Paper**: Multi-Reranker: Maximizing performance of RAG in the FinanceRAG challenge (ACM-ICAIF '24)
- **Authors**: Joohyun Lee, Minji Roh
- **ArXiv**: https://arxiv.org/abs/2411.16732
- **Original repo**: https://github.com/cv-lee/FinanceRAG
- **Key adaptation**: We replace their static finance corpus with live Semantic Scholar API ingestion,
  and apply their multi-model reranking strategy to academic literature search instead of finance docs.
## Demo Queries (for testing + report)
 
1. "reducing cloud infrastructure costs using Kubernetes autoscaling"
2. "large language models for academic paper retrieval"
3. "graph neural networks for fraud detection"
4. "transformer models for information retrieval reranking"
## Claude Code Instructions
 
After writing each file, you MUST:
1. Print a plain English explanation of what the file does and why
2. Explain each function — what it takes as input, what it does, what it returns
3. Explain any non-obvious technical choices (e.g. why BM25, why concatenate title+abstract)
4. Show the exact command to run that file or test it
5. Show expected output so I know it's working correctly
Format each explanation like this:
 
---
### What this file does
...
 
### Function breakdown
- `function_name(args)` — what it does in plain English
### Why we made these choices
...
 
### How to run it
```bash
python filename.py --arg value
```
 
### Expected output
...
---
 
Do this for every file you create, no exceptions.
 
## Dependencies
 
```
requests
rank_bm25
sentence-transformers
pandas
torch
```
 
Install with:
```bash
pip install requests rank_bm25 sentence-transformers pandas torch
```