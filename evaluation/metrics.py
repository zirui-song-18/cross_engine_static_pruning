"""
Evaluation metrics for sparse retrieval experiments.

Computes:
  - Recall@k (oracle-based, against unpruned results)
  - NDCG@k (against human qrels)
  - MRR@k (against human qrels)
  - Success@k / Recall_Judge@k (at least one relevant doc in top-k)
"""

import numpy as np
from pathlib import Path
from typing import Optional


def load_results(result_file: str) -> np.ndarray:
    """Load result array from text file (one row per query, comma-separated doc IDs)."""
    return np.loadtxt(result_file, delimiter=",", dtype="int32")


def load_ground_truth(gt_file: str) -> np.ndarray:
    """Load ground truth array from text file."""
    return np.loadtxt(gt_file, delimiter=",", dtype="int32")


def load_qrels(qrels_file: str, query_ids_file: str):
    """Load qrels as dict: {qid: {doc_id: relevance}}."""
    qrels = {}
    with open(qrels_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, doc_id, rel = parts[0], parts[1], parts[2], int(parts[3])
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][doc_id] = rel

    query_ids = []
    with open(query_ids_file) as f:
        for line in f:
            query_ids.append(line.strip())

    return qrels, query_ids


def recall_at_k(results: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Compute recall@k: fraction of oracle top-k results found in predicted top-k."""
    n_queries = results.shape[0]
    recalls = []
    for i in range(n_queries):
        pred = set(results[i, :k].tolist())
        gt = set(ground_truth[i, :k].tolist())
        if len(gt) == 0:
            continue
        recalls.append(len(pred & gt) / len(gt))
    return np.mean(recalls) if recalls else 0.0


def ndcg_at_k(results: np.ndarray, qrels: dict, query_ids: list, k: int) -> float:
    """Compute NDCG@k using human relevance judgments."""
    ndcgs = []
    for i, qid in enumerate(query_ids):
        if qid not in qrels:
            continue
        rel_map = qrels[qid]
        # DCG
        dcg = 0.0
        for rank in range(min(k, results.shape[1])):
            doc_id = str(results[i, rank])
            rel = rel_map.get(doc_id, 0)
            dcg += (2**rel - 1) / np.log2(rank + 2)
        # IDCG
        sorted_rels = sorted(rel_map.values(), reverse=True)[:k]
        idcg = sum((2**r - 1) / np.log2(rank + 2) for rank, r in enumerate(sorted_rels))
        if idcg > 0:
            ndcgs.append(dcg / idcg)
    return np.mean(ndcgs) if ndcgs else 0.0


def mrr_at_k(results: np.ndarray, qrels: dict, query_ids: list, k: int) -> float:
    """Compute MRR@k (Mean Reciprocal Rank)."""
    rrs = []
    for i, qid in enumerate(query_ids):
        if qid not in qrels:
            continue
        rel_map = qrels[qid]
        for rank in range(min(k, results.shape[1])):
            doc_id = str(results[i, rank])
            if rel_map.get(doc_id, 0) > 0:
                rrs.append(1.0 / (rank + 1))
                break
        else:
            rrs.append(0.0)
    return np.mean(rrs) if rrs else 0.0


def success_at_k(results: np.ndarray, qrels: dict, query_ids: list, k: int) -> float:
    """Compute Success@k (fraction of queries with at least one relevant doc in top-k)."""
    successes = []
    for i, qid in enumerate(query_ids):
        if qid not in qrels:
            continue
        rel_map = qrels[qid]
        found = False
        for rank in range(min(k, results.shape[1])):
            doc_id = str(results[i, rank])
            if rel_map.get(doc_id, 0) > 0:
                found = True
                break
        successes.append(1.0 if found else 0.0)
    return np.mean(successes) if successes else 0.0


def evaluate(result_file: str, ground_truth_file: str, k: int = 10) -> float:
    """Compute recall@k against oracle ground truth."""
    results = load_results(result_file)
    gt = load_ground_truth(ground_truth_file)
    return recall_at_k(results, gt, k)


def evaluate_all_metrics(
    result_file: str,
    ground_truth_file: Optional[str],
    qrels_file: str,
    query_ids_file: str,
    k: int = 10,
):
    """Compute and print all metrics."""
    results = load_results(result_file)

    if ground_truth_file and Path(ground_truth_file).exists():
        gt = load_ground_truth(ground_truth_file)
        rec = recall_at_k(results, gt, k)
        print(f"Recall@{k}: {rec:.4f}")
    else:
        print(f"Recall@{k}: N/A (no ground truth)")

    qrels, query_ids = load_qrels(qrels_file, query_ids_file)
    ndcg = ndcg_at_k(results, qrels, query_ids, k)
    mrr = mrr_at_k(results, qrels, query_ids, k)
    suc = success_at_k(results, qrels, query_ids, k)

    print(f"nDCG@{k}: {ndcg:.4f}")
    print(f"MRR@{k}: {mrr:.4f}")
    print(f"Recall_Judge@{k}: {suc:.4f}")
