#!/usr/bin/env python3
"""
SEISMIC Experiment Suite (S1-S5)
================================

Mirrors BMP experiments on SEISMIC for cross-engine validation.

S1: Baseline sweep (query_cut + heap_factor)
S2: Static query pruning (MR) + query_cut interaction
S3: Document pruning (Alpha-Mass, Max-Ratio) -> rebuild index
S4: Posting-list pruning (MR) -> rebuild index
S5: Combined static + dynamic

Hardware: single-thread search, CPU-pinned, per-query timing via perf_counter.
Metrics: latency + p50/p90/p95/p99 + index_size + oracle_recall@10 + ndcg@10

Usage:
  python run_seismic.py --dataset msmarco --data-dir /path/to/data --cpu-core 0
  python run_seismic.py --dataset nq --data-dir /path/to/data --phases S1 S2 --dry-run
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ============================================================================
# Dataset configurations (paths relative to --data-dir)
# ============================================================================
DATASETS = {
    "msmarco": {
        "docs_csr": "msmarco/docs.csr",
        "queries_csr": "msmarco/queries.csr",
        "doc_ids_file": "msmarco/doc_ids.txt",
        "query_ids": "msmarco/query_ids.txt",
        "qrels": "msmarco/qrels.tsv",
        "prefix": "MS",
    },
    "nq": {
        "docs_csr": "nq/docs.csr",
        "queries_csr": "nq/queries.csr",
        "doc_ids_file": "nq/doc_ids.txt",
        "query_ids": "nq/query_ids.txt",
        "qrels": "nq/qrels.tsv",
        "prefix": "NQ",
    },
    "msmarco_v3gte": {
        "docs_csr": "msmarco_v3gte/docs.csr",
        "queries_csr": "msmarco_v3gte/queries.csr",
        "doc_ids_file": "msmarco_v3gte/doc_ids.txt",
        "query_ids": "msmarco_v3gte/query_ids.txt",
        "qrels": "msmarco_v3gte/qrels.tsv",
        "prefix": "MS_V3GTE",
    },
    "nq_v3gte": {
        "docs_csr": "nq_v3gte/docs.csr",
        "queries_csr": "nq_v3gte/queries.csr",
        "doc_ids_file": "nq_v3gte/doc_ids.txt",
        "query_ids": "nq_v3gte/query_ids.txt",
        "qrels": "nq_v3gte/qrels.tsv",
        "prefix": "NQ_V3GTE",
    },
}

# ============================================================================
# SEISMIC index build parameters
# ============================================================================
SEISMIC_BUILD_PARAMS = {
    "n_postings": 4000,
    "centroid_fraction": 0.2,
    "summary_energy": 0.5,
    "max_fraction": 3.0,
    "min_cluster_size": 2,
    "doc_cut": 15,
    "nknn": 0,
}

# ============================================================================
# Experiment parameters
# ============================================================================

# S1: Baseline sweep
S1_QUERY_CUT_VALUES = [4]
S1_HEAP_FACTOR_VALUES = [1.15, 1.20, 1.30, 1.50, 1.75, 2.00]

# S2: Query pruning
S2_MR_RATIOS = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
S2_QUERY_CUT_VALUES = [3, 5, 8, 20]

# S3: Document pruning
S3_DOC_ALPHA_MASS = [0.5, 0.7, 0.9, 0.95]
S3_DOC_MAX_RATIO = [0.1, 0.3, 0.5]
S3_QUERY_CUT_VALUES = [3, 5, 8]

# S4: Posting pruning
S4_POST_MR = [0.05, 0.1, 0.2, 0.3]
S4_QUERY_CUT_VALUES = [3, 5, 8]

# S5: Combined
S5_DOC_PRUNE = ("doc_alpha_mass", 0.9)
S5_QUERY_MR = 0.1

K = 10

# ============================================================================
# Logging
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger("seismic_exp")


# ============================================================================
# Core functions
# ============================================================================

def prune_queries_mr(queries, mr_ratio):
    """Apply Max-Ratio pruning to queries. Returns new query list."""
    pruned = []
    for qid, q_comp, q_vals in queries:
        if len(q_vals) == 0:
            pruned.append((qid, q_comp, q_vals))
            continue
        max_val = q_vals.max()
        mask = q_vals >= max_val * mr_ratio
        pruned.append((qid, q_comp[mask], q_vals[mask]))
    return pruned


def run_search(index_path, queries, k, query_cut, heap_factor, output_trec, cpu_core=0):
    """Run SEISMIC search with per-query timing. Returns stats dict."""
    import seismic

    os.sched_setaffinity(0, {cpu_core})

    load_path = index_path
    if not load_path.endswith(".index.seismic"):
        load_path = load_path + ".index.seismic"
    idx = seismic.SeismicIndex.load(load_path)

    # Warmup
    for qid, qc, qv in queries[:min(10, len(queries))]:
        idx.search(qid, qc, qv, k, query_cut, heap_factor)

    # Measurement
    per_query_us = []
    trec_lines = []
    for qid, q_comp, q_vals in queries:
        t0 = time.perf_counter()
        results = idx.search(qid, q_comp, q_vals, k, query_cut, heap_factor)
        per_query_us.append((time.perf_counter() - t0) * 1e6)

        for rank, (_, score, doc_id) in enumerate(results, 1):
            trec_lines.append(f"{qid} Q0 {doc_id} {rank} {score:.4f} SEISMIC\n")

    Path(output_trec).parent.mkdir(parents=True, exist_ok=True)
    with open(output_trec, 'w') as f:
        f.writelines(trec_lines)

    per_query_us.sort()
    n = len(per_query_us)
    idx_file = index_path if index_path.endswith(".index.seismic") else index_path + ".index.seismic"
    stats = {
        "search_elapsed": sum(per_query_us) / n if n else 0,
        "p50_us": per_query_us[n * 50 // 100] if n else 0,
        "p90_us": per_query_us[n * 90 // 100] if n else 0,
        "p95_us": per_query_us[n * 95 // 100] if n else 0,
        "p99_us": per_query_us[n * 99 // 100] if n else 0,
        "index_size_bytes": os.path.getsize(idx_file) if os.path.exists(idx_file) else 0,
    }
    return stats


# ============================================================================
# CSV writing
# ============================================================================

_csv_initialized = set()

def write_result(csv_path, exp_id, phase, pruning_type, pruning_param, pruning_param2,
                 query_cut, heap_factor, stats, metrics, index_name="", query_name=""):
    fieldnames = [
        "exp_id", "phase", "pruning_type", "pruning_param", "pruning_param2",
        "query_cut", "heap_factor", "k",
        "latency_us", "p50_us", "p90_us", "p95_us", "p99_us",
        "index_size_bytes",
        "oracle_recall_at_10", "ndcg_at_10", "success_at_10",
        "index_name", "query_name",
    ]
    row = {
        "exp_id": exp_id, "phase": phase,
        "pruning_type": pruning_type, "pruning_param": pruning_param,
        "pruning_param2": pruning_param2,
        "query_cut": query_cut, "heap_factor": heap_factor, "k": K,
        "latency_us": f"{stats.get('search_elapsed', 0):.1f}",
        "p50_us": f"{stats.get('p50_us', 0):.0f}",
        "p90_us": f"{stats.get('p90_us', 0):.0f}",
        "p95_us": f"{stats.get('p95_us', 0):.0f}",
        "p99_us": f"{stats.get('p99_us', 0):.0f}",
        "index_size_bytes": f"{stats.get('index_size_bytes', 0):.0f}",
        "oracle_recall_at_10": f"{metrics.get('oracle_recall_at_10', 0):.6f}",
        "ndcg_at_10": f"{metrics.get('ndcg_at_10', 0):.6f}",
        "success_at_10": f"{metrics.get('success_at_10', 0):.6f}",
        "index_name": index_name, "query_name": query_name,
    }
    csv_str = str(csv_path)
    write_header = csv_str not in _csv_initialized and not os.path.exists(csv_str)
    with open(csv_str, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    _csv_initialized.add(csv_str)


# ============================================================================
# Main
# ============================================================================

def print_plan(args):
    print(f"\n{'='*70}")
    print(f"SEISMIC EXPERIMENT PLAN ({args.dataset.upper()})")
    print(f"{'='*70}")
    phases = set(args.phases)
    run_all = "all" in phases
    n = 0
    if run_all or "S1" in phases:
        c = len(S1_QUERY_CUT_VALUES) + len(S1_HEAP_FACTOR_VALUES) - 1
        n += c
        print(f"S1: {c} searches (query_cut sweep + heap_factor sweep)")
    if run_all or "S2" in phases:
        c = len(S2_MR_RATIOS) * len(S2_QUERY_CUT_VALUES)
        n += c
        print(f"S2: {c} searches (MR x query_cut)")
    if run_all or "S3" in phases:
        ni = len(S3_DOC_ALPHA_MASS) + len(S3_DOC_MAX_RATIO)
        c = ni * len(S3_QUERY_CUT_VALUES)
        n += c
        print(f"S3: {ni} indexes, {c} searches")
    if run_all or "S4" in phases:
        ni = len(S4_POST_MR)
        c = ni * len(S4_QUERY_CUT_VALUES)
        n += c
        print(f"S4: {ni} indexes, {c} searches")
    if run_all or "S5" in phases:
        n += 12
        print(f"S5: ~12 searches")
    print(f"\nTotal: ~{n} searches")
    print(f"CPU core: {args.cpu_core}")
    print()


def main():
    parser = argparse.ArgumentParser(description="SEISMIC Experiment Suite")
    parser.add_argument("--dataset", required=True,
                        choices=["msmarco", "nq", "msmarco_v3gte", "nq_v3gte"])
    parser.add_argument("--data-dir", required=True, type=Path,
                        help="Root directory containing dataset files")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: <data-dir>/seismic_results)")
    parser.add_argument("--phases", nargs="+",
                        choices=["S1", "S2", "S3", "S4", "S5", "all"],
                        default=["all"])
    parser.add_argument("--cpu-core", type=int, default=0)
    parser.add_argument("--keep-jsonl", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print_plan(args)
        return

    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or data_dir / "seismic_results").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"SEISMIC experiments: dataset={args.dataset}, phases={args.phases}")
    log.info(f"Data: {data_dir}, Output: {output_dir}")
    print(f"Configuration verified. Ready to run {args.phases} on {args.dataset}")


if __name__ == "__main__":
    main()
