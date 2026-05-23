# Cross-Engine Static Pruning for Sparse Retrieval

Code repository for the paper: *Static Pruning Portability Across Sparse Retrieval Engines*

## Overview

This paper evaluates static pruning strategies (Alpha-Mass and Max-Ratio) across three fundamentally different sparse retrieval engines, two datasets, and two encoders (1,140 configurations total). We show that:

1. **Pruning strategies are portable** across engine architectures with consistent quality-latency trade-offs
2. **NDCG saturates before recall** at moderate pruning levels, providing a practical operating point
3. **Micro-architectural profiling** reveals the window-switch accumulator achieves significantly lower cache-miss rates

### Engines Evaluated

| Engine | Architecture | Accumulation Strategy |
|--------|-------------|----------------------|
| Custom C++ | Inverted index with window-switch accumulator | Document-ordered, windowed |
| [BMP](https://github.com/JMMackenzie/BMP) | Block-Max Pruning (Rust) | Block-partitioned, dynamic pruning |
| [SEISMIC](https://github.com/TusKANNy/seismic) | Graph-based sparse retrieval | Cluster-navigating |

### Datasets

| Dataset | Documents | Encoder | Avg Query Terms |
|---------|-----------|---------|-----------------|
| MS MARCO Passage | 8.8M | SPLADE-cocondenser | 44 |
| MS MARCO Passage | 8.8M | V3-GTE | 7 |
| Natural Questions | 2.7M | SPLADE-cocondenser | 47 |
| Natural Questions | 2.7M | V3-GTE | 7 |

## Repository Structure

```
.
├── engine/                     # Custom C++ sparse retrieval engine
│   ├── CMakeLists.txt          # Build configuration
│   ├── src/                    # C++ source files
│   │   ├── AlphaMassDoc_AlphaMassPosting.{h,cpp}  # Main search with pruning
│   │   ├── InvertedIndexWindowed.h                 # Window-switch accumulator
│   │   ├── AlphaMassQuery.{h,cpp}                  # Query-only pruning variant
│   │   ├── LinscanIndex.{h,cpp}                    # Baseline linear scan
│   │   ├── CsrReader.{h,cpp}                       # CSR format reader
│   │   ├── py_binding.cpp                          # pybind11 Python interface
│   │   └── ...                                     # Supporting headers
│   ├── pybind11/               # pybind11 submodule (clone separately)
│   └── cereal/                 # cereal serialization (clone separately)
├── scripts/
│   ├── custom/                 # Custom C++ pipeline experiment scripts
│   │   ├── run_experiments.py  # Full pruning experiment suite
│   │   └── bench.py            # Single-run benchmark driver
│   ├── bmp/                    # BMP integration scripts
│   │   ├── fast_search.py      # Fast multi-config BMP search
│   │   └── run_bmp_pruning.py  # B3/B4/B5 index-side pruning
│   ├── seismic/                # SEISMIC integration scripts
│   │   ├── run_seismic.py      # S1-S5 experiment suite
│   │   └── csr_to_seismic.py   # CSR to SEISMIC JSONL converter
│   └── profiling/              # Perf stat profiling scripts
│       └── run_perf.sh
├── evaluation/                 # Quality metric computation
│   ├── __init__.py
│   └── metrics.py              # Recall, NDCG, MRR, Success@k
├── configs/                    # Experiment configuration files
│   ├── experiment_space.json   # Full 1,140-config space definition
│   ├── custom_msmarco.yaml     # Custom engine config example
│   ├── bmp_msmarco.yaml        # BMP config example
│   └── seismic_msmarco.yaml    # SEISMIC config example
├── figures/                    # Output directory for generated figures
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Building the Custom C++ Engine

### Prerequisites

- CMake >= 3.15
- C++20 compiler (GCC 11+ or Clang 14+)
- Python 3.8+ with development headers
- pybind11 (clone into `engine/pybind11/`)

### Build Steps

```bash
# Clone pybind11 into the engine directory
cd engine
git clone https://github.com/pybind/pybind11.git

# Create build directory
mkdir build && cd build

# Configure and build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# The built module (sparse_engine.so) will be in the build/ directory
# Add it to your PYTHONPATH or install via pip
```

### Build Options

- `-DENABLE_FLOAT=ON`: Use float32 values (default: int8 quantized)
- `-DNO_RERANK=ON`: Disable two-stage re-ranking (single-stage mode)

## Reproducing Experiments

### Data Preparation

Experiments require sparse-encoded document and query collections in CSR binary format.
The CSR format stores: `[nrow, ncol, nnz (int64)] [indptr (int64)] [indices (int32)] [data (float32)]`.

You will need:
- Document CSR files from SPLADE or V3-GTE encoding
- Query CSR files (matching encoder)
- Qrels files (TREC format)
- Query ID files (one ID per line)

### Running Custom C++ Experiments

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full experiment suite
python scripts/custom/run_experiments.py \
    --dataset msmarco \
    --data-dir /path/to/your/data \
    --phases all \
    --cpu-core 0

# Dry run to see experiment plan
python scripts/custom/run_experiments.py \
    --dataset msmarco \
    --data-dir /path/to/your/data \
    --dry-run
```

### Running BMP Experiments

Requires the [BMP](https://github.com/JMMackenzie/BMP) repository built from source.

```bash
# Build BMP (Rust)
cd /path/to/BMP
cargo build --release

# Run pruning experiments
python scripts/bmp/run_bmp_pruning.py \
    --bmp-dir /path/to/BMP \
    --data-dir /path/to/your/data \
    --phases B3 B4 B5 \
    --cpu-core 0
```

### Running SEISMIC Experiments

Requires [SEISMIC](https://github.com/TusKANNy/seismic) Python bindings.

```bash
# Install SEISMIC
pip install seismic

# Run experiment suite
python scripts/seismic/run_seismic.py \
    --dataset msmarco \
    --data-dir /path/to/your/data \
    --phases all \
    --cpu-core 0
```

### Generating Figures

```bash
# Set the result directory
export RESULT_DIR=/path/to/experiment/results

# Generate individual figures
python scripts/figures/fig1_portability_pareto.py
python scripts/figures/fig3_seismic_pareto.py
python scripts/figures/fig5_profiling.py --prof-dir /path/to/profiling/results
python scripts/figures/fig6_ndcg_saturation.py --data-dir /path/to/fig6/data
python scripts/figures/fig7_per_query_variance.py --per-query-dir /path/to/per_query/csvs
```

## Measurement Methodology

All latency measurements follow a consistent protocol:

1. **Single-thread execution**: `taskset -c 0` pins search to one CPU core
2. **Warmup**: 10 queries discarded before measurement
3. **Per-query timing**: `perf_counter` (BMP/SEISMIC) or C++ `chrono::high_resolution_clock`
4. **Two-pass**: Warmup run + measurement run (BMP binary)
5. **Metrics reported**: Average, P50, P90, P95, P99 latency

## External Dependencies

| Tool | Version | Purpose | Source |
|------|---------|---------|--------|
| BMP | commit `main` | Block-Max Pruning engine | https://github.com/JMMackenzie/BMP |
| SEISMIC | >= 0.3.0 | Graph-based sparse retrieval | https://github.com/TusKANNy/seismic |
| pybind11 | >= 2.10 | C++/Python bindings | https://github.com/pybind/pybind11 |

## Configuration Space

The full experiment space (1,140 configurations) is defined in `configs/experiment_space.json`.
It covers the Cartesian product of:

- 3 engines (Custom C++, BMP, SEISMIC)
- 4 dataset-encoder combinations
- 6 pruning strategies at multiple intensity levels
- Engine-specific runtime parameters (alpha/beta for BMP, query_cut/heap_factor for SEISMIC)

## License

This code is released for academic review purposes.
