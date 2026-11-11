---
title: PaperRank
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Find papers for a research question and spot gaps
---

# PaperRank

**Try it live:** https://samuelw67-paperrank.hf.space

Searching for papers usually goes like this: you type a question, get fifty
results, and spend the next hour opening abstracts to figure out which ones
actually answer it. PaperRank tries to do that first pass for you.

You ask a research question in plain English. PaperRank searches
[OpenAlex](https://openalex.org) (200M+ papers), reranks the results with a
model that reads your question next to each abstract, and then shows you which
parts of your question each paper covers, with quotes from the abstract as proof.
It also points out combinations that look understudied, which is handy if
you're hunting for a research gap.

I built it for the Forward Data Lab research task at UIUC. The two-stage
"retrieve fast, then rerank carefully" idea comes from
[Multi-Reranker (Lee & Roh, ACM-ICAIF 2024)](https://arxiv.org/abs/2411.16732),
which used it for financial documents. I wanted to see how well it holds up for
academic search.

## How it works

Say you ask *"CRISPR delivery methods for liver diseases in vivo"*.

1. **Break the question down.** An LLM (via Groq) splits it into aspects,
   like *gene editing*, *delivery method*, *target organ* and *in vivo setting*,
   and writes 3–4 search queries that phrase it in different ways.
2. **Search.** Each query runs as a semantic search on OpenAlex. The results
   get merged and de-duplicated, and anything without an abstract is dropped.
3. **Rerank.** A CrossEncoder scores every paper against your original
   question. This is slower than OpenAlex's own ranking but a lot more precise,
   since it reads the question and the abstract together instead of comparing
   two separate embeddings.
4. **Check coverage.** For the top papers, the LLM marks each aspect as
   covered, partial or missing, and has to quote the abstract to back it up.
   If the quote isn't really in the abstract, it gets thrown out. That keeps
   the model honest.
5. **Look for gaps.** PaperRank counts how many papers mention *all* your
   aspects together, then drops one aspect at a time. If "all four" gives 300
   papers but leaving out *gene editing* gives 9,700, that tells you something.

Everything is based on abstracts only, so treat the gaps as leads to check,
not conclusions.

## What's in here

There's no separate backend. `app.py` is the whole web app: Gradio serves the
page and runs the pipeline in the same Python process.

| File | What it does |
|---|---|
| `app.py` | The Gradio web app. Search box, results, coverage bars, year filter, per-paper summaries, plus caching and rate limiting so a public deploy doesn't burn through API budgets. |
| `retriever.py` | Talks to OpenAlex. Runs searches (several at once), rebuilds abstracts from OpenAlex's odd inverted-index format, and counts matching papers for the gap check. |
| `reranker.py` | Loads a CrossEncoder and reranks papers. Every model's scores get squashed to 0–1 so they're comparable. |
| `analyzer.py` | All the LLM parts: breaking down the question, checking coverage, finding gaps, writing summaries. |
| `main.py` | The same pipeline from the command line, if you don't want to open a browser. |
| `evaluate.py` | Offline evaluation (NDCG@10, MRR) comparing plain OpenAlex ranking against multi-query search and reranking. Questions live in `eval/queries.txt`. |
| `requirements.txt` | Python dependencies, pinned to versions I've tested. Pulls the CPU-only build of PyTorch to keep installs small. |

## Reranker models

You can switch models in the sidebar:

- **MiniLM** (`cross-encoder/ms-marco-MiniLM-L-6-v2`): the default. Small
  and fast, and honestly good enough most of the time.
- **BGE v2 m3** (`BAAI/bge-reranker-v2-m3`): the one the Multi-Reranker
  paper used. The most accurate, but much bigger and slower on CPU.
- **Electra** (`cross-encoder/ms-marco-electra-base`): somewhere in between.

Models download from Hugging Face the first time you use them, then stay cached.

## Running it locally

You'll need Python 3.10+ and a free [Groq API key](https://console.groq.com/keys).

```bash
git clone https://github.com/samwjya/PaperRank.git
cd PaperRank
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project folder:

```
GROQ_API_KEY=your_groq_key
OPEN_ALEX_API=your_openalex_key   # optional, but gives you your own daily budget
```

Then start the app and keep the terminal open while you use it:

```bash
python app.py
```

It opens at http://127.0.0.1:7860. A new question takes anywhere from 20
seconds to a minute (most of that is OpenAlex and the reranker). Asking the
same thing again is instant because results are cached for a day.

Or skip the browser:

```bash
python main.py "CRISPR delivery methods for liver diseases in vivo" --top 5
```

### Other settings

These are all optional environment variables:

- `GROQ_MODEL`: which Groq model to use (default `qwen/qwen3.8-27b`)
- `DAILY_SEARCH_LIMIT`: a cap on searches across all users per day (default 100)

## Deploying to Hugging Face Spaces

The live version runs at https://huggingface.co/spaces/samuelw67/PaperRank on
the free CPU tier. To deploy your own copy:

1. Create a new Space at https://huggingface.co/new-space, pick **Gradio**
   as the SDK and **CPU basic (free)** as the hardware.
2. In the Space's **Settings → Variables and secrets**, add `GROQ_API_KEY`
   (and `OPEN_ALEX_API` if you have one) as **secrets**.
3. Push this repo to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/PaperRank
   git push --force space main
   ```
   When git asks for a password, use a Hugging Face access token with write
   access, not your account password.

The block at the top of this README tells Spaces which Gradio and Python
versions to use and to run `app.py`. The first build takes a few minutes,
mostly installing PyTorch. After that, every push rebuilds the Space.

## Evaluation

```bash
python evaluate.py build   # fetch candidates and have an LLM judge relevance (one-time)
python evaluate.py run     # score each ranking method and print NDCG / MRR
```

The relevance labels come from an LLM judge, so they're a starting point. If
you want numbers you can really trust, fix them by hand in `eval/dataset.json`.

## Limitations

- It only sees abstracts, not full papers. If something's only in the methods
  section, PaperRank won't know about it.
- A "gap" can just mean people use different words for the same thing.
- OpenAlex's semantic search returns at most 50 results per query, which is
  why PaperRank runs several queries instead of one big one.
- On a free CPU Space, BGE is slow. Give it a bit.

## Credits

- Lee & Roh, *Multi-Reranker: Maximizing performance of retrieval-augmented
  generation in the FinanceRAG challenge*, ACM-ICAIF 2024
  ([paper](https://arxiv.org/abs/2411.16732), [code](https://github.com/cv-lee/FinanceRAG))
- [OpenAlex](https://openalex.org) for the free scholarly data
- [Groq](https://groq.com) for fast LLM inference
- [sentence-transformers](https://www.sbert.net) for the CrossEncoder models
