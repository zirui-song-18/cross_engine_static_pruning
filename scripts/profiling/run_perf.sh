#!/bin/bash
#
# Comprehensive Perf Profiling: BMP and SEISMIC across pruning levels
# ===================================================================
# Collects cache-miss, IPC, dTLB for BMP and SEISMIC at multiple pruning levels.
#
# Usage:
#   bash run_perf.sh --bmp-index /path/to/index.bmp --bmp-queries /path/to/dev.pisa \
#       --seismic-index /path/to/seismic/index --seismic-queries /path/to/queries.jsonl
#
# Expected runtime: ~30-60 minutes
#

set -e

# Configuration (override via environment or arguments)
CPU_CORE="${CPU_CORE:-0}"
PERF_EVENTS="cycles,instructions,cache-references,cache-misses,dTLB-loads,dTLB-load-misses,branch-instructions,branch-misses"
OUTDIR="${OUTDIR:-perf_results}"
BMP_PYTHON="${BMP_PYTHON:-python3}"
SEISMIC_PYTHON="${SEISMIC_PYTHON:-python3}"

mkdir -p "$OUTDIR"

# ============================================================================
# Helper: run one BMP perf experiment
# ============================================================================
run_bmp_perf() {
    local LABEL="$1"
    local INDEX="$2"
    local QUERIES="$3"
    local ALPHA="$4"
    local BETA="$5"
    local K="${6:-10}"
    local OUTFILE="$OUTDIR/perf_bmp_${LABEL}.txt"

    echo "  [BMP] $LABEL (alpha=$ALPHA, beta=$BETA)"
    taskset -c $CPU_CORE perf stat -e $PERF_EVENTS -- \
        $BMP_PYTHON scripts/profiling/perf_bmp_search.py "$INDEX" "$QUERIES" "$ALPHA" "$BETA" "$K" \
        > "$OUTFILE" 2>&1
    QPS=$(grep "^QPS:" "$OUTFILE" | awk '{print $2}')
    CM=$(grep "cache-misses" "$OUTFILE" | grep -oP '[\d.]+(?=\s*%)')
    IPC=$(grep "insn per cycle" "$OUTFILE" | grep -oP '[\d.]+(?=\s+insn)')
    DTLB=$(grep "dTLB-load-misses" "$OUTFILE" | grep -oP '[\d.]+(?=%)')
    printf "    QPS=%-8s IPC=%-5s Cache-Miss=%-6s%% dTLB-Miss=%-6s%%\n" "$QPS" "$IPC" "$CM" "$DTLB"
}

# ============================================================================
# Helper: run one SEISMIC perf experiment
# ============================================================================
run_seismic_perf() {
    local LABEL="$1"
    local INDEX="$2"
    local QUERIES="$3"
    local QC="$4"
    local HF="$5"
    local K="${6:-10}"
    local OUTFILE="$OUTDIR/perf_seismic_${LABEL}.txt"

    echo "  [SEISMIC] $LABEL (qc=$QC, hf=$HF)"
    taskset -c $CPU_CORE perf stat -e $PERF_EVENTS -- \
        $SEISMIC_PYTHON scripts/profiling/perf_seismic_search.py "$INDEX" "$QUERIES" "$QC" "$HF" "$K" \
        > "$OUTFILE" 2>&1
    QPS=$(grep "^QPS:" "$OUTFILE" | awk '{print $2}')
    CM=$(grep "cache-misses" "$OUTFILE" | grep -oP '[\d.]+(?=\s*%)')
    IPC=$(grep "insn per cycle" "$OUTFILE" | grep -oP '[\d.]+(?=\s+insn)')
    DTLB=$(grep "dTLB-load-misses" "$OUTFILE" | grep -oP '[\d.]+(?=%)')
    printf "    QPS=%-8s IPC=%-5s Cache-Miss=%-6s%% dTLB-Miss=%-6s%%\n" "$QPS" "$IPC" "$CM" "$DTLB"
}

# ============================================================================
echo "============================================"
echo "  Comprehensive Perf Profiling"
echo "  $(date)"
echo "  CPU Core: $CPU_CORE"
echo "============================================"
echo ""
echo "Usage: Set environment variables BMP_INDEX, BMP_QUERIES, SEISMIC_INDEX, SEISMIC_QUERIES"
echo "       then run this script."
echo ""

# Example sweep (uncomment and configure paths):
# run_bmp_perf "baseline_beta1.0" "$BMP_INDEX" "$BMP_QUERIES" 1.0 1.0
# run_bmp_perf "baseline_beta0.7" "$BMP_INDEX" "$BMP_QUERIES" 1.0 0.7
# run_seismic_perf "baseline_qc5" "$SEISMIC_INDEX" "$SEISMIC_QUERIES" 5 1.0
# run_seismic_perf "baseline_qc3" "$SEISMIC_INDEX" "$SEISMIC_QUERIES" 3 1.0

# ============================================================================
# Summary table
# ============================================================================
echo ""
echo "============================================"
echo "  RESULTS"
echo "  $(date)"
echo "============================================"
echo ""
printf "%-30s %8s %5s %8s %8s\n" "Experiment" "QPS" "IPC" "CacheMiss%" "dTLB%"
printf "%-30s %8s %5s %8s %8s\n" "------------------------------" "--------" "-----" "--------" "--------"
for f in "$OUTDIR"/perf_*.txt; do
    [ -f "$f" ] || continue
    NAME=$(basename "$f" .txt | sed 's/perf_//')
    QPS=$(grep "^QPS:" "$f" 2>/dev/null | awk '{print $2}')
    CM=$(grep "cache-misses" "$f" 2>/dev/null | grep -oP '[\d.]+(?=\s*%)' | head -1)
    IPC=$(grep "insn per cycle" "$f" 2>/dev/null | grep -oP '[\d.]+(?=\s+insn)' | head -1)
    DTLB=$(grep "dTLB-load-misses" "$f" 2>/dev/null | grep -oP '[\d.]+(?=%)' | head -1)
    [ -z "$QPS" ] && QPS="err"
    printf "%-30s %8s %5s %8s %8s\n" "$NAME" "$QPS" "$IPC" "$CM" "$DTLB"
done
