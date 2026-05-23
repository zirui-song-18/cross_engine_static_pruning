#!/usr/bin/env python3
"""
Custom C++ Pipeline Experiments
================================

Runs pruning experiments on the custom InvertedIndexWindowed pipeline
with per-query latency output, proper metrics collection, and structured CSV.

Supports datasets:
  - msmarco: MS MARCO Passage (8.8M docs, SPLADE)
  - nq: Natural Questions (2.7M docs, SPLADE)
  - msmarco_v3gte: MS MARCO (V3-GTE encoder, ~7 query terms)
  - nq_v3gte: Natural Questions (V3-GTE encoder)

Experiments:
  Q:  Query pruning (Alpha-Mass sweep)
  D:  Document pruning (Alpha-Mass + Max-Ratio)
  P:  Posting-list pruning (Alpha-Mass + Max-Ratio)
  QD: Combined query + doc pruning
  QP: Combined query + posting pruning

Collects:
  - Latency: avg, p50, p90, p95, p99
  - Per-query latency CSV
  - Memory: inverted index size, forward index size
  - Quality: Recall@k, NDCG@k, MRR@k, Recall_Judge@k

Usage:
  python run_experiments.py --dataset msmarco --data-dir /path/to/data
  python run_experiments.py --dataset nq --data-dir /path/to/data
  python run_experiments.py --dataset msmarco --phases Q D --dry-run
  python run_experiments.py --dataset msmarco --k 1000 --kprime 1000
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# CPU core for taskset pinning (set from --cpu-core arg in main)
CPU_CORE = 0

# ============================================================================
# Dataset configurations (paths relative to --data-dir)
# ============================================================================
DATASETS = {
    "msmarco": {
        "doc_file": "msmarco/docs.csr",
        "query_file": "msmarco/queries.csr",
        "ground_truth_file": "msmarco/ground_truth.txt",
        "qrels_file": "msmarco/qrels.tsv",
        "query_ids_file": "msmarco/query_ids.txt",
        "prefix": "MS",
        "num_worker": 1,
    },
    "nq": {
        "doc_file": "nq/docs.csr",
        "query_file": "nq/queries.csr",
        "ground_truth_file": "nq/ground_truth.txt",
        "qrels_file": "nq/qrels.tsv",
        "query_ids_file": "nq/query_ids.txt",
        "prefix": "NQ",
        "num_worker": 1,
    },
    "msmarco_v3gte": {
        "doc_file": "msmarco_v3gte/docs.csr",
        "query_file": "msmarco_v3gte/queries.csr",
        "ground_truth_file": "msmarco_v3gte/ground_truth.txt",
        "qrels_file": "msmarco_v3gte/qrels.tsv",
        "query_ids_file": "msmarco_v3gte/query_ids.txt",
        "prefix": "MS_V3GTE",
        "num_worker": 1,
    },
    "nq_v3gte": {
        "doc_file": "nq_v3gte/docs.csr",
        "query_file": "nq_v3gte/queries.csr",
        "ground_truth_file": "nq_v3gte/ground_truth.txt",
        "qrels_file": "nq_v3gte/qrels.tsv",
        "query_ids_file": "nq_v3gte/query_ids.txt",
        "prefix": "NQ_V3GTE",
        "num_worker": 1,
    },
}

# ============================================================================
# Experiment parameters (matching BMP/SEISMIC for cross-engine comparison)
# ============================================================================

# Query pruning: Alpha-Mass sweep
Q_ALPHA_MASS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
# Query pruning: Max-Ratio sweep
Q_MAX_RATIO = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

# Doc pruning: Alpha-Mass
D_ALPHA_MASS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
# Doc pruning: Max-Ratio
D_MAX_RATIO = [0.1, 0.2, 0.3, 0.4, 0.5]

# Posting-list pruning: Alpha-Mass
P_ALPHA_MASS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
# Posting-list pruning: Max-Ratio
P_MAX_RATIO = [0.05, 0.1, 0.2, 0.3]

# Combined: representative points
COMBINED_Q_ALPHA = [0.5, 0.7, 0.9]
COMBINED_D_ALPHA = [0.9]
COMBINED_P_ALPHA = [0.9]


# ============================================================================
# Core: run one experiment config and parse output
# ============================================================================

def run_experiment(
    dataset_name: str,
    exp_id: str,
    data_dir: Path,
    bench_py: Path,
    doc_alpha: float = 0.0,
    list_alpha: float = 0.0,
    doc_max_ratio: float = -1.0,
    doc_fixed_top: int = -1,
    list_max_ratio: float = -1.0,
    list_fixed_top: int = -1,
    idf_prune: float = 0.0,
    query_alpha: float = 0.0,
    query_max_ratio: float = -1.0,
    query_fixed_top: int = -1,
    k: int = 10,
    kprime: int = 50,
) -> dict:
    """Run a single experiment via bench.py, parse stdout, collect metrics."""

    ds = DATASETS[dataset_name]
    result_dir = data_dir / "results" / dataset_name
    result_dir.mkdir(parents=True, exist_ok=True)
    per_query_dir = result_dir / "per_query"
    per_query_dir.mkdir(parents=True, exist_ok=True)

    output_log = result_dir / f"{exp_id}.txt"
    per_query_csv = str(per_query_dir / f"{exp_id}_latencies.csv")

    gt_file = str(data_dir / ds["ground_truth_file"])

    cmd = [
        sys.executable, str(bench_py),
        "alpha-mass-doc-alpha-posting-list",
        "--doc-file", str(data_dir / ds["doc_file"]),
        "--query-file", str(data_dir / ds["query_file"]),
        "--ground-truth-file", gt_file,
        "--qrels-file", str(data_dir / ds["qrels_file"]),
        "--query-ids-file", str(data_dir / ds["query_ids_file"]),
        "--doc-alpha-prune-ratio", str(doc_alpha),
        "--list-alpha-prune-ratio", str(list_alpha),
        "--idf-prune-percent", str(idf_prune),
        "--doc-max-ratio", str(doc_max_ratio),
        "--doc-fixed-top", str(doc_fixed_top),
        "--list-max-ratio", str(list_max_ratio),
        "--list-fixed-top", str(list_fixed_top),
        "--k", str(k),
        "--kprime", str(kprime),
        "--query-alpha-prune-ratio", str(query_alpha),
        "--query-max-ratio", str(query_max_ratio),
        "--query-fixed-top", str(query_fixed_top),
        "--num-worker", str(ds["num_worker"]),
        "--per-query-output", per_query_csv,
    ]

    env = os.environ.copy()
    env["ANALYSIS_DEBUG"] = "1"

    # Pin to single CPU core, consistent with BMP/SEISMIC
    cmd = ["taskset", "-c", str(CPU_CORE)] + cmd

    print(f"  [{exp_id}] Running (single-thread, core {CPU_CORE})...")
    t0 = time.time()
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=7200,
        cwd=str(data_dir), env=env,
    )
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"  [{exp_id}] FAILED ({elapsed:.0f}s)")
        print(f"    stderr: {proc.stderr[:500]}")
        with open(str(output_log), 'w') as f:
            f.write(proc.stdout + "\n---STDERR---\n" + proc.stderr)
        return None

    with open(str(output_log), 'w') as f:
        f.write(proc.stdout)

    metrics = parse_experiment_output(proc.stdout, exp_id)
    if metrics:
        metrics["elapsed_s"] = elapsed
        print(f"  [{exp_id}] Done ({elapsed:.0f}s): "
              f"lat={metrics.get('avg_latency_ms', 0):.1f}ms | "
              f"R@10={metrics.get('recall_at_10', 0):.4f} | "
              f"NDCG={metrics.get('ndcg_at_10', 0):.4f} | "
              f"inv={metrics.get('inverted_mb', 0):.0f}MB")
    return metrics


def parse_experiment_output(stdout: str, exp_id: str) -> dict:
    """Parse bench.py stdout to extract all metrics."""
    metrics = {"exp_id": exp_id}

    for line in stdout.split('\n'):
        m = re.search(r'Inverted index: ([\d.]+) MB, Forward index: ([\d.]+) MB', line)
        if m:
            metrics["inverted_mb"] = float(m.group(1))
            metrics["forward_mb"] = float(m.group(2))

        m = re.search(r'QPS:([\d.]+), average latency per query:([\d.]+) ms.*?min:([\d.]+) ms.*?max:([\d.]+) ms.*?median:([\d.]+) ms.*?p95:([\d.]+) ms.*?p99:([\d.]+) ms', line)
        if m:
            metrics["qps"] = float(m.group(1))
            metrics["avg_latency_ms"] = float(m.group(2))
            metrics["min_latency_ms"] = float(m.group(3))
            metrics["max_latency_ms"] = float(m.group(4))
            metrics["median_latency_ms"] = float(m.group(5))
            metrics["p95_latency_ms"] = float(m.group(6))
            metrics["p99_latency_ms"] = float(m.group(7))

        m = re.search(r'Recall@10: ([\d.]+)', line)
        if m:
            metrics["recall_at_10"] = float(m.group(1))

        m = re.search(r'nDCG@10: ([\d.]+)', line)
        if m:
            metrics["ndcg_at_10"] = float(m.group(1))

        m = re.search(r'MRR@10: ([\d.]+)', line)
        if m:
            metrics["mrr_at_10"] = float(m.group(1))

        m = re.search(r'Recall_Judge@10: ([\d.]+)', line)
        if m:
            metrics["recall_judge_at_10"] = float(m.group(1))

        m = re.search(r'average inverted_index_length: (\d+)', line)
        if m:
            metrics["avg_inverted_index_length"] = int(m.group(1))

    return metrics


# ============================================================================
# CSV result writing
# ============================================================================

_csv_init = set()

def write_csv(csv_path, metrics, pruning_config):
    """Write one row to CSV."""
    fieldnames = [
        "exp_id", "phase", "pruning_type",
        "query_alpha", "query_max_ratio",
        "doc_alpha", "doc_max_ratio",
        "list_alpha", "list_max_ratio",
        "k", "kprime",
        "avg_latency_ms", "median_latency_ms", "p95_latency_ms", "p99_latency_ms",
        "min_latency_ms", "max_latency_ms", "qps",
        "inverted_mb", "forward_mb",
        "recall_at_10", "ndcg_at_10", "mrr_at_10", "recall_judge_at_10",
        "avg_inverted_index_length",
    ]
    row = {**pruning_config}
    row.update({k: metrics.get(k, "") for k in fieldnames if k not in row})

    csv_str = str(csv_path)
    write_header = csv_str not in _csv_init and not os.path.exists(csv_str)
    with open(csv_str, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    _csv_init.add(csv_str)


# ============================================================================
# Experiment phases
# ============================================================================

def run_phase_Q(dataset_name, csv_path, data_dir, bench_py, k, kprime):
    """Query pruning: Alpha-Mass and Max-Ratio sweeps."""
    print("=" * 60)
    print(f"PHASE Q: Query Pruning ({dataset_name.upper()})")
    print("=" * 60)

    for alpha in Q_ALPHA_MASS:
        exp_id = f"{DATASETS[dataset_name]['prefix']}_Q_am{alpha:.2f}"
        config = {
            "exp_id": exp_id, "phase": "Q", "pruning_type": "query_alpha_mass",
            "query_alpha": alpha, "query_max_ratio": -1,
            "doc_alpha": 0.0, "doc_max_ratio": -1,
            "list_alpha": 0.0, "list_max_ratio": -1,
            "k": k, "kprime": kprime,
        }
        metrics = run_experiment(dataset_name, exp_id, data_dir, bench_py,
                                 query_alpha=alpha, k=k, kprime=kprime)
        if metrics:
            write_csv(csv_path, metrics, config)

    for mr in Q_MAX_RATIO:
        exp_id = f"{DATASETS[dataset_name]['prefix']}_Q_mr{mr:.2f}"
        config = {
            "exp_id": exp_id, "phase": "Q", "pruning_type": "query_max_ratio",
            "query_alpha": 0.0, "query_max_ratio": mr,
            "doc_alpha": 0.0, "doc_max_ratio": -1,
            "list_alpha": 0.0, "list_max_ratio": -1,
            "k": k, "kprime": kprime,
        }
        metrics = run_experiment(dataset_name, exp_id, data_dir, bench_py,
                                 query_max_ratio=mr, k=k, kprime=kprime)
        if metrics:
            write_csv(csv_path, metrics, config)


def run_phase_D(dataset_name, csv_path, data_dir, bench_py, k, kprime):
    """Document pruning: Alpha-Mass and Max-Ratio sweeps."""
    print("=" * 60)
    print(f"PHASE D: Document Pruning ({dataset_name.upper()})")
    print("=" * 60)

    for alpha in D_ALPHA_MASS:
        exp_id = f"{DATASETS[dataset_name]['prefix']}_D_am{alpha:.2f}"
        config = {
            "exp_id": exp_id, "phase": "D", "pruning_type": "doc_alpha_mass",
            "query_alpha": 0.0, "query_max_ratio": -1,
            "doc_alpha": alpha, "doc_max_ratio": -1,
            "list_alpha": 0.0, "list_max_ratio": -1,
            "k": k, "kprime": kprime,
        }
        metrics = run_experiment(dataset_name, exp_id, data_dir, bench_py,
                                 doc_alpha=alpha, k=k, kprime=kprime)
        if metrics:
            write_csv(csv_path, metrics, config)

    for mr in D_MAX_RATIO:
        exp_id = f"{DATASETS[dataset_name]['prefix']}_D_mr{mr:.2f}"
        config = {
            "exp_id": exp_id, "phase": "D", "pruning_type": "doc_max_ratio",
            "query_alpha": 0.0, "query_max_ratio": -1,
            "doc_alpha": 0.0, "doc_max_ratio": mr,
            "list_alpha": 0.0, "list_max_ratio": -1,
            "k": k, "kprime": kprime,
        }
        metrics = run_experiment(dataset_name, exp_id, data_dir, bench_py,
                                 doc_max_ratio=mr, k=k, kprime=kprime)
        if metrics:
            write_csv(csv_path, metrics, config)


def run_phase_P(dataset_name, csv_path, data_dir, bench_py, k, kprime):
    """Posting-list pruning: Alpha-Mass and Max-Ratio sweeps."""
    print("=" * 60)
    print(f"PHASE P: Posting-List Pruning ({dataset_name.upper()})")
    print("=" * 60)

    for alpha in P_ALPHA_MASS:
        exp_id = f"{DATASETS[dataset_name]['prefix']}_P_am{alpha:.2f}"
        config = {
            "exp_id": exp_id, "phase": "P", "pruning_type": "posting_alpha_mass",
            "query_alpha": 0.0, "query_max_ratio": -1,
            "doc_alpha": 0.0, "doc_max_ratio": -1,
            "list_alpha": alpha, "list_max_ratio": -1,
            "k": k, "kprime": kprime,
        }
        metrics = run_experiment(dataset_name, exp_id, data_dir, bench_py,
                                 list_alpha=alpha, k=k, kprime=kprime)
        if metrics:
            write_csv(csv_path, metrics, config)

    for mr in P_MAX_RATIO:
        exp_id = f"{DATASETS[dataset_name]['prefix']}_P_mr{mr:.2f}"
        config = {
            "exp_id": exp_id, "phase": "P", "pruning_type": "posting_max_ratio",
            "query_alpha": 0.0, "query_max_ratio": -1,
            "doc_alpha": 0.0, "doc_max_ratio": -1,
            "list_alpha": 0.0, "list_max_ratio": mr,
            "k": k, "kprime": kprime,
        }
        metrics = run_experiment(dataset_name, exp_id, data_dir, bench_py,
                                 list_max_ratio=mr, k=k, kprime=kprime)
        if metrics:
            write_csv(csv_path, metrics, config)


def run_phase_QD(dataset_name, csv_path, data_dir, bench_py, k, kprime):
    """Combined: query + document pruning."""
    print("=" * 60)
    print(f"PHASE QD: Combined Query+Doc ({dataset_name.upper()})")
    print("=" * 60)

    for qa in COMBINED_Q_ALPHA:
        for da in COMBINED_D_ALPHA:
            exp_id = f"{DATASETS[dataset_name]['prefix']}_QD_qa{qa:.2f}_da{da:.2f}"
            config = {
                "exp_id": exp_id, "phase": "QD", "pruning_type": "combined_query_doc",
                "query_alpha": qa, "query_max_ratio": -1,
                "doc_alpha": da, "doc_max_ratio": -1,
                "list_alpha": 0.0, "list_max_ratio": -1,
                "k": k, "kprime": kprime,
            }
            metrics = run_experiment(dataset_name, exp_id, data_dir, bench_py,
                                     query_alpha=qa, doc_alpha=da, k=k, kprime=kprime)
            if metrics:
                write_csv(csv_path, metrics, config)


def run_phase_QP(dataset_name, csv_path, data_dir, bench_py, k, kprime):
    """Combined: query + posting-list pruning."""
    print("=" * 60)
    print(f"PHASE QP: Combined Query+Posting ({dataset_name.upper()})")
    print("=" * 60)

    for qa in COMBINED_Q_ALPHA:
        for pa in COMBINED_P_ALPHA:
            exp_id = f"{DATASETS[dataset_name]['prefix']}_QP_qa{qa:.2f}_pa{pa:.2f}"
            config = {
                "exp_id": exp_id, "phase": "QP", "pruning_type": "combined_query_posting",
                "query_alpha": qa, "query_max_ratio": -1,
                "doc_alpha": 0.0, "doc_max_ratio": -1,
                "list_alpha": pa, "list_max_ratio": -1,
                "k": k, "kprime": kprime,
            }
            metrics = run_experiment(dataset_name, exp_id, data_dir, bench_py,
                                     query_alpha=qa, list_alpha=pa, k=k, kprime=kprime)
            if metrics:
                write_csv(csv_path, metrics, config)


# ============================================================================
# Main
# ============================================================================

def print_plan(args):
    ds = args.dataset.upper()
    phases = set(args.phases)
    run_all = "all" in phases
    n = 0
    print(f"\n{'='*60}")
    print(f"CUSTOM PIPELINE EXPERIMENT PLAN ({ds})")
    print(f"{'='*60}")
    print(f"k={args.k}, kprime={args.kprime}")
    if run_all or "Q" in phases:
        c = len(Q_ALPHA_MASS) + len(Q_MAX_RATIO)
        n += c
        print(f"Q (Query pruning):   {c} experiments")
    if run_all or "D" in phases:
        c = len(D_ALPHA_MASS) + len(D_MAX_RATIO)
        n += c
        print(f"D (Doc pruning):     {c} experiments")
    if run_all or "P" in phases:
        c = len(P_ALPHA_MASS) + len(P_MAX_RATIO)
        n += c
        print(f"P (Posting pruning): {c} experiments")
    if run_all or "QD" in phases:
        c = len(COMBINED_Q_ALPHA) * len(COMBINED_D_ALPHA)
        n += c
        print(f"QD (Combined Q+D):   {c} experiments")
    if run_all or "QP" in phases:
        c = len(COMBINED_Q_ALPHA) * len(COMBINED_P_ALPHA)
        n += c
        print(f"QP (Combined Q+P):   {c} experiments")
    print(f"\nTotal: {n} experiments")
    print()


def main():
    parser = argparse.ArgumentParser(description="Custom C++ Pipeline Experiments")
    parser.add_argument("--dataset", required=True,
                        choices=["msmarco", "nq", "msmarco_v3gte", "nq_v3gte"])
    parser.add_argument("--data-dir", required=True, type=Path,
                        help="Root directory containing dataset files")
    parser.add_argument("--bench-py", type=Path, default=None,
                        help="Path to bench.py (default: ../bench.py relative to this script)")
    parser.add_argument("--phases", nargs="+",
                        choices=["Q", "D", "P", "QD", "QP", "all"],
                        default=["all"])
    parser.add_argument("--k", type=int, default=10, help="Top-k (default: 10)")
    parser.add_argument("--kprime", type=int, default=50, help="Rerank pool (default: 50)")
    parser.add_argument("--cpu-core", type=int, default=0, help="CPU core to pin (default: 0)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    global CPU_CORE
    CPU_CORE = args.cpu_core

    if args.dry_run:
        print_plan(args)
        return

    data_dir = args.data_dir.resolve()
    bench_py = args.bench_py or (Path(__file__).parent / "bench.py")

    # Verify data files exist
    ds = DATASETS[args.dataset]
    for key in ["doc_file", "query_file", "qrels_file", "query_ids_file"]:
        p = data_dir / ds[key]
        if not p.exists():
            print(f"ERROR: Missing {key}: {p}")
            sys.exit(1)

    csv_path = data_dir / "results" / args.dataset / f"custom_{args.dataset}_metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    phases = set(args.phases)
    run_all = "all" in phases
    t0 = time.time()

    if run_all or "Q" in phases:
        run_phase_Q(args.dataset, csv_path, data_dir, bench_py, args.k, args.kprime)
    if run_all or "D" in phases:
        run_phase_D(args.dataset, csv_path, data_dir, bench_py, args.k, args.kprime)
    if run_all or "P" in phases:
        run_phase_P(args.dataset, csv_path, data_dir, bench_py, args.k, args.kprime)
    if run_all or "QD" in phases:
        run_phase_QD(args.dataset, csv_path, data_dir, bench_py, args.k, args.kprime)
    if run_all or "QP" in phases:
        run_phase_QP(args.dataset, csv_path, data_dir, bench_py, args.k, args.kprime)

    print(f"\n{'='*60}")
    print(f"ALL DONE in {(time.time()-t0)/3600:.1f}h")
    print(f"Results: {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
