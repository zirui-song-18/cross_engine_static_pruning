#!/usr/bin/env python3
"""
Convert CSR sparse matrices to SEISMIC JSONL format.

Supports optional document-level and posting-level pruning before conversion.

SEISMIC JSONL format (one JSON per line):
  {"id": "0", "vector": {"token1": 2.33, "token2": 1.65, ...}}

Usage:
  # Full corpus (no pruning)
  python csr_to_seismic.py --docs docs.csr --vocab vocab.txt --doc-ids doc_ids.txt --output docs.jsonl

  # With document pruning (Alpha-Mass)
  python csr_to_seismic.py --docs docs.csr --vocab vocab.txt --doc-ids doc_ids.txt \
      --output docs_am0.9.jsonl --doc-prune-alpha 0.9

  # With document pruning (Max-Ratio)
  python csr_to_seismic.py --docs docs.csr --vocab vocab.txt --doc-ids doc_ids.txt \
      --output docs_mr0.3.jsonl --doc-prune-max-ratio 0.3

  # With posting-list pruning (Max-Ratio)
  python csr_to_seismic.py --docs docs.csr --vocab vocab.txt --doc-ids doc_ids.txt \
      --output docs_post0.1.jsonl --posting-prune-max-ratio 0.1

  # Convert queries
  python csr_to_seismic.py --queries queries.csr --vocab vocab.txt --query-ids query_ids.txt \
      --output queries.jsonl
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger("csr2seismic")


def read_csr(path: str):
    """Read CSR binary file -> (data, indices, indptr, ncol)."""
    log.info(f"Reading CSR: {path}")
    with open(path, "rb") as f:
        sizes = np.fromfile(f, dtype="int64", count=3)
        nrow, ncol, nnz = sizes
        indptr = np.fromfile(f, dtype="int64", count=nrow + 1)
        indices = np.fromfile(f, dtype="int32", count=nnz)
        data = np.fromfile(f, dtype="float32", count=nnz)
    log.info(f"  {nrow:,} rows x {ncol:,} cols, {nnz:,} nnz")
    return data, indices, indptr, int(nrow), int(ncol)


def load_vocab(path: str):
    log.info(f"Loading vocab: {path}")
    with open(path) as f:
        vocab = [line.strip() for line in f]
    log.info(f"  {len(vocab)} tokens")
    return vocab


def load_ids(path: str):
    with open(path) as f:
        return [line.strip() for line in f]


def apply_doc_pruning(doc_indices, doc_data, alpha_mass=None, max_ratio=None):
    """Apply document-level pruning to one document's terms."""
    if len(doc_data) == 0:
        return doc_indices, doc_data
    if max_ratio is not None:
        threshold = doc_data.max() * max_ratio
        mask = doc_data >= threshold
        doc_indices, doc_data = doc_indices[mask], doc_data[mask]
        if len(doc_data) == 0:
            return doc_indices, doc_data
    if alpha_mass is not None:
        order = np.argsort(-doc_data)
        sorted_data = doc_data[order]
        cumsum = np.cumsum(sorted_data)
        cutoff = np.searchsorted(cumsum, alpha_mass * cumsum[-1]) + 1
        doc_indices = doc_indices[order[:cutoff]]
        doc_data = sorted_data[:cutoff]
    return doc_indices, doc_data


def convert_docs(
    csr_path, vocab_path, doc_ids_path, output_path,
    doc_prune_alpha=None, doc_prune_max_ratio=None,
    posting_prune_max_ratio=None,
):
    """Convert document CSR to SEISMIC JSONL with optional pruning."""
    data, indices, indptr, nrow, ncol = read_csr(csr_path)
    vocab = load_vocab(vocab_path)
    doc_ids = load_ids(doc_ids_path)

    need_doc_prune = doc_prune_alpha is not None or doc_prune_max_ratio is not None
    need_posting_prune = posting_prune_max_ratio is not None

    if need_posting_prune:
        log.info(f"Applying posting-level MR pruning (ratio={posting_prune_max_ratio})...")
        from scipy.sparse import csr_matrix
        mat = csr_matrix((data, indices, indptr), shape=(nrow, ncol))
        csc = mat.tocsc()
        keep_mask = np.ones(len(csc.data), dtype=bool)
        for term_id in range(ncol):
            s, e = csc.indptr[term_id], csc.indptr[term_id + 1]
            if s == e:
                continue
            term_scores = csc.data[s:e]
            threshold = term_scores.max() * posting_prune_max_ratio
            keep_mask[s:e] = term_scores >= threshold
        pruned_count = int(np.sum(~keep_mask))
        log.info(f"  Removed {pruned_count:,} postings ({pruned_count/len(data)*100:.1f}%)")
        csc.data = csc.data[keep_mask]
        csc.indices = csc.indices[keep_mask]
        new_indptr = np.zeros_like(csc.indptr)
        for term_id in range(ncol):
            s, e = csc.indptr[term_id], csc.indptr[term_id + 1]
            new_indptr[term_id + 1] = new_indptr[term_id] + int(np.sum(keep_mask[s:e]))
        csc.indptr = new_indptr
        mat2 = csc.tocsr()
        data, indices, indptr = mat2.data, mat2.indices, mat2.indptr
        del mat, csc, mat2

    log.info(f"Writing JSONL to {output_path}...")
    t0 = time.time()
    total_terms = 0
    with open(output_path, 'w') as f:
        for doc_id in range(nrow):
            s, e = int(indptr[doc_id]), int(indptr[doc_id + 1])
            doc_idx = indices[s:e]
            doc_dat = data[s:e]

            if need_doc_prune:
                doc_idx, doc_dat = apply_doc_pruning(
                    doc_idx, doc_dat, doc_prune_alpha, doc_prune_max_ratio)

            vector = {}
            for tid, val in zip(doc_idx, doc_dat):
                if val > 0 and tid < len(vocab):
                    vector[vocab[tid]] = round(float(val), 6)
            total_terms += len(vector)

            obj = {"id": doc_ids[doc_id], "vector": vector}
            f.write(json.dumps(obj) + '\n')

            if (doc_id + 1) % 500000 == 0:
                log.info(f"  {doc_id+1:,}/{nrow:,} docs ({(doc_id+1)/nrow*100:.0f}%)")

    elapsed = time.time() - t0
    log.info(f"Done: {nrow:,} docs, {total_terms:,} total terms, {elapsed:.0f}s")
    log.info(f"  Avg terms/doc: {total_terms/nrow:.1f}")


def convert_queries(csr_path, vocab_path, query_ids_path, output_path):
    """Convert query CSR to SEISMIC JSONL."""
    data, indices, indptr, nrow, ncol = read_csr(csr_path)
    vocab = load_vocab(vocab_path)
    query_ids = load_ids(query_ids_path)

    log.info(f"Writing {nrow} queries to {output_path}...")
    with open(output_path, 'w') as f:
        for qi in range(nrow):
            s, e = int(indptr[qi]), int(indptr[qi + 1])
            vector = {}
            for tid, val in zip(indices[s:e], data[s:e]):
                if val > 0 and tid < len(vocab):
                    vector[vocab[tid]] = round(float(val), 6)
            obj = {"id": query_ids[qi], "vector": vector}
            f.write(json.dumps(obj) + '\n')
    log.info(f"Done: {nrow} queries")


def main():
    parser = argparse.ArgumentParser(description="CSR to SEISMIC JSONL converter")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--docs", help="Document CSR file")
    group.add_argument("--queries", help="Query CSR file")
    parser.add_argument("--vocab", required=True, help="Vocab file (one token per line)")
    parser.add_argument("--doc-ids", help="Document IDs file")
    parser.add_argument("--query-ids", help="Query IDs file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--doc-prune-alpha", type=float, help="Alpha-Mass doc pruning")
    parser.add_argument("--doc-prune-max-ratio", type=float, help="Max-Ratio doc pruning")
    parser.add_argument("--posting-prune-max-ratio", type=float, help="Max-Ratio posting pruning")
    args = parser.parse_args()

    if args.docs:
        convert_docs(args.docs, args.vocab, args.doc_ids, args.output,
                     args.doc_prune_alpha, args.doc_prune_max_ratio,
                     args.posting_prune_max_ratio)
    else:
        convert_queries(args.queries, args.vocab, args.query_ids, args.output)


if __name__ == "__main__":
    main()
