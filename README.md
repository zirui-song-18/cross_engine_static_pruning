# Cross-Engine Static Pruning for Sparse Retrieval

Code repository for the paper: *Static Pruning Across Sparse Retrieval Regimes: What Transfers, What Breaks, and What Still Helps* (CIKM '26).

## Overview

This paper evaluates static pruning strategies (Alpha-Mass and Max-Ratio) across three fundamentally different sparse retrieval engines, two datasets, and two encoders (1,140 configurations total). We show that:

1. **Pruning strategies are portable** across engine architectures with consistent quality-latency trade-offs
2. **NDCG saturates before recall** at moderate pruning levels, providing a practical operating point
3. **Micro-architectural profiling** reveals the window-switch accumulator achieves significantly lower cache-miss rates

### Engines Evaluated

| Engine | Architecture | Accumulation Strategy |
|--------|-------------|----------------------|
| Custom C++ | Inverted index with window-switch accumulator | Document-ordered, windowed |
| [BMP](https://github.com/pisa-engine/BMP) | Block-Max Pruning (Rust) | Block-partitioned, dynamic pruning |
| [SEISMIC](https://github.com/TusKANNy/seismic) | Clustered inverted index | Cluster-navigating |

### Datasets

| Dataset | Documents | Encoder | Avg Query Terms |
|---------|-----------|---------|-----------------|
| MS MARCO Passage | 8.8M | SPLADE-cocondenser | 44 |
| MS MARCO Passage | 8.8M | V3-GTE | 7 |
| Natural Questions | 2.7M | SPLADE-cocondenser | 47 |
| Natural Questions | 2.7M | V3-GTE | 7 |

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

Requires the [BMP](https://github.com/pisa-engine/BMP) repository built from source.

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
| BMP | commit `main` | Block-Max Pruning engine | https://github.com/pisa-engine/BMP |
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

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
