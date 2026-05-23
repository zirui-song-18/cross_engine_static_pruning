#!/usr/bin/env python3
"""
Benchmark driver for the custom C++ sparse retrieval engine.

Provides CLI commands for running various search configurations with
different pruning strategies. Measures latency, memory, and quality metrics.

Usage:
  python bench.py alpha-mass-doc-alpha-posting-list --doc-file docs.csr \
      --query-file queries.csr --doc-alpha-prune-ratio 0.9 --k 10
"""

import sys
import os

try:
    import sparse_engine
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../engine/build"))
    import sparse_engine

# Add evaluation module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from scipy.sparse import csr_matrix, vstack
import numpy as np
import time
import click
import os
from evaluation import evaluate, evaluate_all_metrics


@click.group()
def cli():
    """Main command group"""
    pass


def read_sparse_matrix_fields(fname):
    """Read the fields of a CSR matrix without instantiating it."""
    with open(fname, "rb") as f:
        sizes = np.fromfile(f, dtype="int64", count=3)
        nrow, ncol, nnz = sizes
        indptr = np.fromfile(f, dtype="int64", count=nrow + 1)
        assert nnz == indptr[-1]
        indices = np.fromfile(f, dtype="int32", count=nnz)
        assert np.all(indices >= 0) and np.all(indices < ncol)
        data = np.fromfile(f, dtype="float32", count=nnz)
        return data, indices, indptr, ncol


def read_sparse_matrix(fname):
    """Read a CSR matrix in spmat format."""
    data, indices, indptr, ncol = read_sparse_matrix_fields(fname)
    return csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, ncol))


def measure_time(func):
    """A decorator to measure running time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time of {func.__name__}: {end_time - start_time} seconds")
        return result
    return wrapper


def measure_query_time(func):
    """A decorator to measure running time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        query_count = args[0].shape[0]
        print(
            f"Execution time of {func.__name__}: {end_time - start_time} seconds, QPS: {query_count / (end_time - start_time)}"
        )
        return result
    return wrapper


@measure_query_time
def search(X, pc_index, search_parameter):
    return pc_index.search(X.shape, X.indptr, X.indices, X.data, search_parameter)


def get_data_file(filename):
    if os.path.isabs(filename):
        return filename
    return os.path.join(os.path.dirname(__file__), "data", filename)


@measure_time
def build_index(index, filepath):
    index.load(get_data_file(filepath), 10000)


def quantize_to_uint8(sparse_matrix):
    data = sparse_matrix.data
    max_val = 3.0
    clamped_data = np.clip(data, 0, max_val)
    scaled_data = (clamped_data / max_val * 255).round().astype(np.uint8)
    return csr_matrix(
        (scaled_data, sparse_matrix.indices, sparse_matrix.indptr),
        shape=sparse_matrix.shape,
    )


def output_memory(text):
    import psutil
    memory = psutil.virtual_memory()
    print(f"{text}: {memory.percent}%\n")


@cli.command()
@click.option("--ip-budget", default=0, help="time budget (microseconds), 0=no limit")
@click.option("--alpha-prune-ratio", default=0.5, help="alpha mass pruning for docs")
@click.option("--list-alpha-prune-ratio", default=-1.0, help="alpha mass pruning for posting lists")
@click.option("--idf-prune-percent", default=-1.0, help="percent of longest posting lists to remove")
@click.option("--doc-max-ratio", default=-1.0, help="doc prune: keep weights > max*ratio")
@click.option("--doc-fixed-top", default=-1, help="doc prune: keep top N weights")
@click.option("--list-max-ratio", default=-1.0, help="posting list prune: keep weights > max*ratio")
@click.option("--list-fixed-top", default=-1, help="posting list prune: keep top N weights")
@click.option("--doc-file", default="docs.csr", help="document CSR file")
@click.option("--query-file", default="queries.csr", help="query CSR file")
@click.option("--ground-truth-file", default="ground_truth.txt", help="ground truth file")
@click.option("--qrels-file", default="qrels.tsv", help="qrels file")
@click.option("--query-ids-file", default="query_ids.txt", help="query IDs file")
@click.option("--k", default=10, help="top-k results (default: 10)")
@click.option("--kprime", default=50, help="rerank pool size (default: 50)")
@click.option("--doc-alpha-prune-ratio", default=0.0, help="doc-level alpha-mass pruning (0=none)")
@click.option("--query-alpha-prune-ratio", default=0.0, help="query-level alpha-mass pruning (0=none)")
@click.option("--query-max-ratio", default=-1.0, help="query prune: keep weights > max*ratio")
@click.option("--query-fixed-top", default=-1, help="query prune: keep top N weights")
@click.option("--num-worker", default=8, help="number of search workers (default: 8)")
@click.option("--per-query-output", default="", help="path to write per-query latency CSV")
def alpha_mass_doc_alpha_posting_list(
    ip_budget, alpha_prune_ratio, list_alpha_prune_ratio,
    idf_prune_percent, doc_max_ratio, doc_fixed_top,
    list_max_ratio, list_fixed_top, doc_file, query_file,
    ground_truth_file, qrels_file, query_ids_file,
    k, kprime, doc_alpha_prune_ratio, query_alpha_prune_ratio,
    query_max_ratio, query_fixed_top, num_worker, per_query_output,
):
    """Run linear scan search with various pruning options.

    Doc-level alpha pruning: use --doc-alpha-prune-ratio.
    Query-level alpha pruning: use --query-alpha-prune-ratio.
    These are independent and can be combined.
    """
    effective_doc_alpha = doc_alpha_prune_ratio

    output_memory("before")
    X = read_sparse_matrix(query_file)
    index = sparse_engine.AlphaMassDoc_AlphaMassPosting()
    index.load_alpha_mass(
        doc_file, 10000, effective_doc_alpha, list_alpha_prune_ratio,
        idf_prune_percent, doc_max_ratio, doc_fixed_top,
        list_max_ratio, list_fixed_top,
    )
    inv_mem, fwd_mem = index.get_memory_usage()
    print(
        f"Inverted index: {inv_mem / 1024 / 1024:.2f} MB, Forward index: {fwd_mem / 1024 / 1024:.2f} MB"
    )
    output_memory("memory after load")

    print(f"doc: alpha={effective_doc_alpha}, max_ratio={doc_max_ratio}, fixed_top={doc_fixed_top}")
    print(f"list: alpha={list_alpha_prune_ratio}, max_ratio={list_max_ratio}, fixed_top={list_fixed_top}, idf={idf_prune_percent}")
    print(f"query: alpha={query_alpha_prune_ratio}, max_ratio={query_max_ratio}, fixed_top={query_fixed_top}")

    search_parameter = sparse_engine.QueryArguments(
        k, kprime, 0, 0, ip_budget, 0, num_worker,
        query_alpha_prune_ratio, query_max_ratio, query_fixed_top
    )
    ret = search(X, index, search_parameter)
    output_memory("memory after search")

    if per_query_output:
        index.write_per_query_latencies(per_query_output)

    R = np.array(ret, dtype="int32")
    output = f"linscan_{os.path.basename(doc_file)}_array.txt"
    np.savetxt(output, R, delimiter=",", fmt="%d")

    effective_gt = ground_truth_file
    if not ground_truth_file or ground_truth_file == "/dev/null" or not os.path.exists(ground_truth_file) or os.path.getsize(ground_truth_file) == 0:
        effective_gt = None
    evaluate_all_metrics(output, effective_gt, qrels_file, query_ids_file, k)


if __name__ == "__main__":
    cli()
