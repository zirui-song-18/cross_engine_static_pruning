#!/usr/bin/env python3
"""
BMP Index-Side Pruning Experiments (B3/B4/B5)
=============================================

Pipeline for each pruning config:
  1. CSR -> pruned CIFF  (csr_to_bmp.py with pruning flags)
  2. CIFF -> BMP index   (ciff2bmp)
  3. Delete CIFF         (save disk)
  4. Search              (Rust CLI, taskset -c 0, single-thread)
  5. Evaluate            (Oracle Recall@10, NDCG@10, Success@10)

B3: Document pruning   -- Alpha-Mass x 6, Max-Ratio x 5 = 11 indexes
B4: Posting pruning    -- MR x 7 = 7 indexes
B5: Combined           -- reuse B3 indexes + pruned queries

Usage:
  python run_bmp_pruning.py --bmp-dir /path/to/BMP --data-dir /path/to/data
  python run_bmp_pruning.py --phases B3 --dry-run
  python run_bmp_pruning.py --phases B4 B5 --keep-ciff --cpu-core 4
"""

import argparse
import csv
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================================
# Experiment parameters
# ============================================================================
SEARCH_ALPHAS = [0.5, 0.7, 0.9, 1.0]

B3_DOC_ALPHA_MASS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
B3_DOC_MAX_RATIO = [0.1, 0.2, 0.3, 0.4, 0.5]
B4_POST_MR = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]

B5_DOC_PRUNE = "doc_am0.90"
B5_QUERY_MR = 0.1

BMP_BLOCK_SIZE = 8

# ============================================================================
# Logging
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger("b345")


# ============================================================================
# Core pipeline functions
# ============================================================================

def check_disk_space(path: str, required_gb: float = 20.0):
    """Check if enough disk space is available."""
    stat = shutil.disk_usage(path)
    avail_gb = stat.free / (1024**3)
    if avail_gb < required_gb:
        log.error(f"Insufficient disk space: {avail_gb:.1f}GB available, need {required_gb:.1f}GB")
        return False
    return True


def generate_ciff(output_ciff, label, bmp_dir, docs_csr, vocab, doc_ids, extra_args=None):
    """Generate CIFF from docs.csr via csr_to_bmp.py."""
    if os.path.exists(output_ciff) and os.path.getsize(output_ciff) > 0:
        log.info(f"  [{label}] CIFF exists: {os.path.basename(output_ciff)}")
        return True

    csr_to_bmp = str(Path(bmp_dir) / "csr_to_bmp.py")
    cmd = [
        sys.executable, csr_to_bmp, "docs",
        "--input", str(docs_csr),
        "--output", output_ciff,
        "--vocab", str(vocab),
        "--doc-ids", str(doc_ids),
    ]
    if extra_args:
        cmd.extend(extra_args)

    log.info(f"  [{label}] Generating CIFF...")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, cwd=str(bmp_dir))
    elapsed = time.time() - t0

    if proc.returncode != 0:
        log.error(f"  [{label}] CIFF generation FAILED ({elapsed:.0f}s)")
        return False

    size_gb = os.path.getsize(output_ciff) / (1024**3)
    log.info(f"  [{label}] CIFF done: {size_gb:.1f}GB in {elapsed:.0f}s")
    return True


def build_bmp_index(input_ciff, output_bmp, label, ciff2bmp_bin):
    """Build BMP index from CIFF via ciff2bmp."""
    if os.path.exists(output_bmp) and os.path.getsize(output_bmp) > 0:
        log.info(f"  [{label}] BMP index exists: {os.path.basename(output_bmp)}")
        return True

    cmd = [
        str(ciff2bmp_bin),
        "-c", input_ciff,
        "-o", output_bmp,
        "-b", str(BMP_BLOCK_SIZE),
        "--compress-range",
    ]
    log.info(f"  [{label}] Building BMP index...")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        log.error(f"  [{label}] BMP build FAILED ({elapsed:.0f}s)")
        return False

    size_gb = os.path.getsize(output_bmp) / (1024**3)
    log.info(f"  [{label}] BMP index done: {size_gb:.1f}GB in {elapsed:.0f}s")
    return True


def run_search(search_bin, index_path, query_path, k, alpha, beta, output_trec, cpu_core=0):
    """Run BMP search with Rust CLI + taskset. Returns (success, stats_dict)."""
    Path(output_trec).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "taskset", "-c", str(cpu_core),
        str(search_bin),
        "--index", index_path,
        "--queries", query_path,
        "--k", str(k),
        "--alpha", str(alpha),
        "--beta", str(beta),
    ]

    stats = {}
    for run_idx in range(2):  # run 0 = warmup, run 1 = measurement
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0:
            log.error(f"    Search FAILED: {proc.stderr[:300]}")
            return False, {}
        if run_idx == 1:
            # Parse BMP's internal timing from stderr
            stats = _parse_bmp_stderr(proc.stderr)
            with open(output_trec, 'w') as f:
                f.write(proc.stdout)

    return True, stats


def _parse_bmp_stderr(stderr):
    """Parse BMP search binary stderr for timing stats."""
    import re
    stats = {}
    for line in stderr.split('\n'):
        m = re.search(r'avg_query_time_us:\s*([\d.]+)', line)
        if m:
            stats['search_elapsed'] = float(m.group(1))
        m = re.search(r'p50:\s*([\d.]+)', line)
        if m:
            stats['p50_us'] = float(m.group(1))
        m = re.search(r'p90:\s*([\d.]+)', line)
        if m:
            stats['p90_us'] = float(m.group(1))
        m = re.search(r'p95:\s*([\d.]+)', line)
        if m:
            stats['p95_us'] = float(m.group(1))
        m = re.search(r'p99:\s*([\d.]+)', line)
        if m:
            stats['p99_us'] = float(m.group(1))
    return stats


# ============================================================================
# Result writing
# ============================================================================

_csv_initialized = False

def write_result(csv_path, exp_id, phase, pruning_type, pruning_param, pruning_param2,
                 alpha, beta, stats, metrics, index_path, query_path):
    global _csv_initialized
    fieldnames = [
        "exp_id", "phase", "pruning_type", "pruning_param", "pruning_param2",
        "alpha", "beta", "k",
        "latency_us", "p50_us", "p90_us", "p95_us", "p99_us",
        "index_size_bytes",
        "oracle_recall_at_10", "ndcg_at_10", "success_at_10",
        "index_path", "query_path",
    ]
    row = {
        "exp_id": exp_id, "phase": phase,
        "pruning_type": pruning_type, "pruning_param": pruning_param,
        "pruning_param2": pruning_param2,
        "alpha": alpha, "beta": beta, "k": 10,
        "latency_us": f"{stats.get('search_elapsed', 0):.1f}",
        "p50_us": f"{stats.get('p50_us', 0):.0f}",
        "p90_us": f"{stats.get('p90_us', 0):.0f}",
        "p95_us": f"{stats.get('p95_us', 0):.0f}",
        "p99_us": f"{stats.get('p99_us', 0):.0f}",
        "index_size_bytes": f"{stats.get('index_size_bytes', 0):.0f}",
        "oracle_recall_at_10": f"{metrics.get('oracle_recall_at_10', 0):.6f}",
        "ndcg_at_10": f"{metrics.get('ndcg_at_10', 0):.6f}",
        "success_at_10": f"{metrics.get('success_at_10', 0):.6f}",
        "index_path": os.path.basename(index_path),
        "query_path": os.path.basename(query_path),
    }

    csv_str = str(csv_path)
    write_header = not _csv_initialized and not os.path.exists(csv_str)
    with open(csv_str, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    _csv_initialized = True


# ============================================================================
# Main
# ============================================================================

def print_plan(args):
    print(f"\n{'='*70}")
    print("B3/B4/B5 EXPERIMENT PLAN")
    print(f"{'='*70}")
    print(f"CPU core: {args.cpu_core}")
    print(f"Search alphas: {SEARCH_ALPHAS}")
    phases = set(args.phases)
    n_search = 0
    if "B3" in phases:
        n = len(B3_DOC_ALPHA_MASS) + len(B3_DOC_MAX_RATIO)
        print(f"\nB3: {n} indexes x {len(SEARCH_ALPHAS)} alphas = {n * len(SEARCH_ALPHAS)} searches")
        n_search += n * len(SEARCH_ALPHAS)
    if "B4" in phases:
        n = len(B4_POST_MR)
        print(f"B4: {n} indexes x {len(SEARCH_ALPHAS)} alphas = {n * len(SEARCH_ALPHAS)} searches")
        n_search += n * len(SEARCH_ALPHAS)
    if "B5" in phases:
        print(f"B5: ~{2 * len(SEARCH_ALPHAS)} searches (combined)")
        n_search += 2 * len(SEARCH_ALPHAS)
    print(f"\nTotal: ~{n_search} searches")


def main():
    parser = argparse.ArgumentParser(description="BMP B3/B4/B5 Experiments")
    parser.add_argument("--bmp-dir", required=True, type=Path,
                        help="Path to BMP repository root")
    parser.add_argument("--data-dir", required=True, type=Path,
                        help="Path to data directory (docs.csr, queries.csr, etc.)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: <data-dir>/bmp_results)")
    parser.add_argument("--phases", nargs="+", choices=["B3", "B4", "B5"],
                        default=["B3", "B4", "B5"])
    parser.add_argument("--cpu-core", type=int, default=0)
    parser.add_argument("--keep-ciff", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print_plan(args)
        return

    bmp_dir = args.bmp_dir.resolve()
    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or data_dir / "bmp_results").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    search_bin = bmp_dir / "target/release/search"
    ciff2bmp_bin = bmp_dir / "target/release/ciff2bmp"

    for p, name in [(search_bin, "BMP search binary"), (ciff2bmp_bin, "ciff2bmp binary")]:
        if not p.exists():
            log.error(f"Missing: {name} at {p}")
            sys.exit(1)

    log.info(f"BMP experiments: phases={args.phases}, cpu_core={args.cpu_core}")
    log.info(f"Output: {output_dir}")

    # Placeholder: actual phase implementations follow the same pattern
    # as the full experiment script (generate CIFF, build index, search, evaluate)
    print(f"Configuration verified. Ready to run {args.phases} on {data_dir}")


if __name__ == "__main__":
    main()
