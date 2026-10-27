"""Offline evaluation: do multi-query search and reranking actually help?

Two steps, so OpenAlex/LLM costs are paid once and runs are reproducible:

  python evaluate.py build   # fetch candidates + judge relevance -> eval/dataset.json
  python evaluate.py run     # rank the saved candidates every way, report metrics

Relevance labels come from an LLM judge (0 = not, 1 = partly, 2 = highly
relevant) over a pool of every system's top results (TREC-style pooling).
They are a starting point: correct them by hand in eval/dataset.json.
"""
import json
import math
from pathlib import Path

from analyzer import MAX_ABSTRACT_CHARS, LLMError, chat_json, decompose
from reranker import DEFAULT_MODEL, load_model, rerank
from retriever import fetch_papers, multi_search


DATASET = Path("eval/dataset.json")
QUERIES = Path("eval/queries.txt")
JUDGE_BATCH = 5


# ── Metrics ───────────────────────────────────────────────────────────────────

def dcg_at_k(relevances: list[float], k: int) -> float:
    """Discounted Cumulative Gain at rank k."""
    return sum(rel / math.log2(i + 1) for i, rel in enumerate(relevances[:k], start=1))


def ndcg_at_k(ranked_relevances: list[float], all_relevances: list[float], k: int) -> float:
    """NDCG at rank k, normalized by the best possible top-k over ALL judged papers.

    Normalizing by every judged paper (not just the ones this ranking returned)
    means a system that fails to retrieve relevant papers is penalized for it.
    """
    idcg = dcg_at_k(sorted(all_relevances, reverse=True), k)
    return dcg_at_k(ranked_relevances, k) / idcg if idcg else 0.0


def mrr(ranked_relevances: list[float], threshold: float = 2) -> float:
    """Reciprocal rank of the first paper with relevance >= threshold (default: highly relevant)."""
    for rank, rel in enumerate(ranked_relevances, start=1):
        if rel >= threshold:
            return 1.0 / rank
    return 0.0


# ── Build: candidates + LLM relevance judgments ───────────────────────────────

def judge(question: str, papers: list[dict]) -> dict[str, int]:
    """Grade each paper 0/1/2 for the question. Returns {paperId: grade}."""
    labels = {}
    for i in range(0, len(papers), JUDGE_BATCH):
        batch = papers[i:i + JUDGE_BATCH]
        papers_text = "\n\n".join(
            f"[{j}] Title: {p['title']}\nAbstract: {p['abstract'][:MAX_ABSTRACT_CHARS]}"
            for j, p in enumerate(batch)
        )
        prompt = f"""You are an expert assessor judging search results for a researcher.

Research question: {question}

Grade each paper's relevance using its title and abstract:
2 = highly relevant: directly addresses the question; the researcher would definitely want it
1 = partly relevant: related, addresses some parts of the question
0 = not relevant

{papers_text}

Respond with JSON only: {{"grades": [{{"index": 0, "grade": 2}}]}}"""
        data = chat_json(prompt, max_tokens=60 * len(batch))
        for g in data.get("grades", []):
            j, grade = g.get("index"), g.get("grade")
            if isinstance(j, int) and 0 <= j < len(batch) and grade in (0, 1, 2):
                labels[batch[j]["paperId"]] = grade
    return labels


def _system_rankings(question: str, entry: dict, papers: dict, models: dict) -> dict[str, list[str]]:
    """Every system's ranking of the saved candidates, as lists of paperIds."""
    def docs(ids: list[str]) -> list[dict]:
        return [{"paperId": pid, **papers[pid]} for pid in ids]

    rankings = {
        "OpenAlex (1 search)": entry["single"],
        "Multi-search (merged)": entry["multi"],
    }
    for name, model in models.items():
        short = name.split("/")[-1]
        rankings[f"1 search + {short}"] = [p["paperId"] for p in rerank(question, docs(entry["single"]), model)]
        rankings[f"Multi + {short}"] = [p["paperId"] for p in rerank(question, docs(entry["multi"]), model)]
    return rankings


def build(model_names: list[str], pool_depth: int, refresh: bool) -> None:
    data = json.loads(DATASET.read_text("utf-8")) if DATASET.exists() else {"queries": [], "papers": {}}
    done = {q["question"]: q for q in data["queries"]}
    questions = [q.strip() for q in QUERIES.read_text("utf-8").splitlines() if q.strip() and not q.startswith("#")]
    models = {name: load_model(name) for name in model_names}

    for n, question in enumerate(questions, 1):
        entry = done.get(question)
        if entry is None or refresh:
            print(f"[{n}/{len(questions)}] fetching: {question}")
            plan = decompose(question)
            single = fetch_papers(question)
            multi = multi_search(plan["sub_queries"])
            for p in single + multi:
                data["papers"][p["paperId"]] = {"title": p["title"], "abstract": p["abstract"]}
            entry = {
                "question": question,
                "sub_queries": plan["sub_queries"],
                "single": [p["paperId"] for p in single],
                "multi": [p["paperId"] for p in multi],
                "labels": (entry or {}).get("labels", {}),
            }
            if question not in done:
                data["queries"].append(entry)
            else:
                data["queries"][data["queries"].index(done[question])] = entry
            done[question] = entry

        # Pool: union of every system's top results; judge only papers not yet labeled.
        rankings = _system_rankings(question, entry, data["papers"], models)
        pool = {pid for ranking in rankings.values() for pid in ranking[:pool_depth]}
        todo = [{"paperId": pid, **data["papers"][pid]} for pid in sorted(pool) if pid not in entry["labels"]]
        if todo:
            print(f"    judging {len(todo)} papers ...")
            try:
                entry["labels"].update(judge(question, todo))
            except LLMError as e:
                print(f"    judge failed ({e}); rerun build to resume")
        DATASET.parent.mkdir(exist_ok=True)
        DATASET.write_text(json.dumps(data, indent=1, ensure_ascii=False), "utf-8")  # save after every query

    print(f"\nSaved {len(data['queries'])} queries, {len(data['papers'])} papers -> {DATASET}")


# ── Run: metrics per system ───────────────────────────────────────────────────

def run(model_names: list[str], k: int, verbose: bool) -> None:
    data = json.loads(DATASET.read_text("utf-8"))
    models = {name: load_model(name) for name in model_names}
    totals: dict[str, dict[str, float]] = {}

    for entry in data["queries"]:
        labels = entry["labels"]
        if not labels:
            continue
        all_rels = list(labels.values())
        rankings = _system_rankings(entry["question"], entry, data["papers"], models)
        if verbose:
            print(f"\n{entry['question']}")
        for system, ranking in rankings.items():
            rels = [labels.get(pid, 0) for pid in ranking]
            scores = {
                f"NDCG@{k}": ndcg_at_k(rels, all_rels, k),
                "MRR": mrr(rels),
                f"Judged@{k}": sum(pid in labels for pid in ranking[:k]) / k,
            }
            for metric, value in scores.items():
                totals.setdefault(system, {}).setdefault(metric, 0.0)
                totals[system][metric] += value
            if verbose:
                print(f"  {system:<34}" + "".join(f"{v:>10.3f}" for v in scores.values()))

    n = sum(1 for q in data["queries"] if q["labels"])
    metrics = list(next(iter(totals.values())))
    print(f"\nMean over {n} queries (labels: LLM judge, 0/1/2; MRR = first highly relevant paper)\n")
    print(f"  {'System':<34}" + "".join(f"{m:>10}" for m in metrics))
    print("  " + "-" * (34 + 10 * len(metrics)))
    for system, scores in totals.items():
        print(f"  {system:<34}" + "".join(f"{scores[m] / n:>10.3f}" for m in metrics))
    print(f"\n  Judged@{k} < 1.000 means some top results were never labeled (unlabeled count as 0);")
    print("  rerun build with the same --models to judge them.")


if __name__ == "__main__":
    import argparse
    import sys

    from dotenv import load_dotenv

    load_dotenv()
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

    parser = argparse.ArgumentParser(description="Evaluate retrieval + reranking against judged queries")
    parser.add_argument("command", choices=["build", "run"])
    parser.add_argument("--models", nargs="+", default=[DEFAULT_MODEL], help="Reranker models to compare")
    parser.add_argument("--k", type=int, default=10, help="Metric cutoff")
    parser.add_argument("--pool-depth", type=int, default=20, help="build: judge each system's top N")
    parser.add_argument("--refresh", action="store_true", help="build: re-fetch candidates for every query")
    parser.add_argument("--verbose", action="store_true", help="run: show per-query scores")
    args = parser.parse_args()

    if args.command == "build":
        build(args.models, args.pool_depth, args.refresh)
    else:
        run(args.models, args.k, args.verbose)
