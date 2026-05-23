#!/usr/bin/env python3
"""
Fast BMP search using Python bindings.

Loads the BMP index ONCE and runs all queries with multiple alpha/beta
configurations. This avoids the ~80s index loading overhead per configuration.

Usage:
  python fast_search.py --index /path/to/index.bmp --queries /path/to/dev.pisa \
      --configs configs.json --output-dir results/trec/ --cpu-core 0
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import bmp


def parse_pisa_queries(pisa_path: str) -> Tuple[List[str], List[Dict[str, float]]]:
    """Parse .pisa query file into (query_ids, [{token: weight}, ...]).

    In .pisa format, tokens are repeated proportionally to their quantized weights.
    We reconstruct weights from repetition counts.
    """
    query_ids = []
    query_dicts = []

    with open(pisa_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            colon_idx = line.index(':')
            qid = line[:colon_idx].strip()
            tokens_str = line[colon_idx + 1:].strip()
            query_ids.append(qid)

            if not tokens_str:
                query_dicts.append({})
                continue

            tokens = tokens_str.split()
            counts = Counter(tokens)
            query_dicts.append({tok: float(cnt) for tok, cnt in counts.items()})

    return query_ids, query_dicts


def run_sweep(
    index_path: str,
    query_ids: List[str],
    query_dicts: List[Dict[str, float]],
    configs: List[dict],
    output_dir: str,
    cpu_core: int = None,
) -> List[dict]:
    """Run multiple (alpha, beta, k) configs with a single index load.

    Each config dict has: exp_id, alpha, beta, k

    Returns list of result dicts with: exp_id, latency_us, trec_path
    """
    if cpu_core is not None:
        os.sched_setaffinity(0, {cpu_core})

    print(f"Loading index: {index_path}")
    t0 = time.time()
    searcher = bmp.Searcher(index_path)
    load_time = time.time() - t0
    print(f"Index loaded in {load_time:.1f}s")

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for cfg in configs:
        exp_id = cfg["exp_id"]
        alpha = cfg["alpha"]
        beta = cfg["beta"]
        k = cfg["k"]
        trec_path = os.path.join(output_dir, f"{exp_id}.trec")

        if os.path.exists(trec_path) and os.path.getsize(trec_path) > 0:
            print(f"  [SKIP] {exp_id}: already exists")
            t_start = time.time()
            for qid, qdict in zip(query_ids[:10], query_dicts[:10]):
                if qdict:
                    searcher.search(qdict, k, alpha, beta)
            t_sample = (time.time() - t_start) / 10 * 1e6
            results.append({
                "exp_id": exp_id, "latency_us": t_sample,
                "trec_path": trec_path,
            })
            continue

        print(f"  [RUN]  {exp_id}: alpha={alpha}, beta={beta}, k={k}")
        trec_lines = []
        total_us = 0

        for qi, (qid, qdict) in enumerate(zip(query_ids, query_dicts)):
            if not qdict:
                continue
            t_start = time.perf_counter()
            doc_ids, scores = searcher.search(qdict, k, alpha, beta)
            t_end = time.perf_counter()
            total_us += (t_end - t_start) * 1e6

            for rank, (did, score) in enumerate(zip(doc_ids, scores), 1):
                trec_lines.append(f"{qid} Q0 {did} {rank} {score:.1f} BMP\n")

        avg_us = total_us / len(query_ids) if query_ids else 0
        print(f"         latency={avg_us:.0f}us/query ({total_us/1e6:.1f}s total)")

        with open(trec_path, 'w') as f:
            f.writelines(trec_lines)

        results.append({
            "exp_id": exp_id, "latency_us": avg_us,
            "trec_path": trec_path,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Fast BMP sweep with single index load")
    parser.add_argument("--index", required=True, help="BMP index path")
    parser.add_argument("--queries", required=True, help=".pisa query file")
    parser.add_argument("--configs", required=True,
                        help="JSON file with list of {exp_id, alpha, beta, k}")
    parser.add_argument("--output-dir", required=True, help="Output directory for TREC files")
    parser.add_argument("--cpu-core", type=int, help="CPU core to pin to")
    args = parser.parse_args()

    with open(args.configs) as f:
        configs = json.load(f)
    print(f"Loaded {len(configs)} experiment configs")

    print(f"Parsing queries from {args.queries}")
    query_ids, query_dicts = parse_pisa_queries(args.queries)
    print(f"Loaded {len(query_ids)} queries")

    results = run_sweep(
        args.index, query_ids, query_dicts,
        configs, args.output_dir, args.cpu_core,
    )

    summary_path = os.path.join(args.output_dir, "sweep_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    print(f"Total experiments: {len(results)}")


if __name__ == "__main__":
    main()
