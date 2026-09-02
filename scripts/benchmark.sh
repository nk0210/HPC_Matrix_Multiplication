#!/usr/bin/env bash
#
# benchmark.sh - drive ./matrix_multiply --benchmark over a grid of matrix
# sizes and thread counts and collect the results into results/benchmark.csv.
#
# For every (matrix_size, threads) combination the benchmark binary is run
# RUNS times.  Each run appends one row to the CSV:
#
#   matrix_size,threads,run,time_seq,time_par,speedup,efficiency,correct
#
# Defaults reproduce Section 8 of the report:
#   sizes   = 500 1000 2000
#   threads = 1 2 4 8
#   runs    = 3
#
# Any of these can be overridden from the environment, which is handy for a
# quick end-to-end check on a slow machine, e.g.:
#
#   MM_SIZES="200 400" MM_THREADS="1 2 4" MM_RUNS=1 bash scripts/benchmark.sh

set -euo pipefail

read -r -a SIZES   <<< "${MM_SIZES:-500 1000 2000}"
read -r -a THREADS <<< "${MM_THREADS:-1 2 4 8}"
RUNS="${MM_RUNS:-3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

BIN="$ROOT_DIR/matrix_multiply"
if [[ ! -x "$BIN" && -x "$BIN.exe" ]]; then
    BIN="$BIN.exe"
fi
if [[ ! -x "$BIN" && ! -x "$BIN.exe" ]]; then
    echo "error: benchmark binary not found at $BIN" >&2
    echo "       build it first with 'make' (or 'mingw32-make')." >&2
    exit 1
fi

RESULTS_DIR="$ROOT_DIR/results"
OUT="$RESULTS_DIR/benchmark.csv"
mkdir -p "$RESULTS_DIR"

echo "matrix_size,threads,run,time_seq,time_par,speedup,efficiency,correct" > "$OUT"

echo "Benchmark grid:"
echo "  sizes   : ${SIZES[*]}"
echo "  threads : ${THREADS[*]}"
echo "  runs    : $RUNS"
echo "  output  : $OUT"
echo

for N in "${SIZES[@]}"; do
    for T in "${THREADS[@]}"; do
        for R in $(seq 1 "$RUNS"); do
            line="$("$BIN" --benchmark "$N" "$T")"
            # line = N,T,time_seq,time_par,speedup,efficiency,correct
            IFS=',' read -r csv_n csv_t t_seq t_par sp eff corr <<< "$line"
            echo "$csv_n,$csv_t,$R,$t_seq,$t_par,$sp,$eff,$corr" >> "$OUT"
            printf 'size=%-5s threads=%-2s run=%-2s -> speedup=%s efficiency=%s%% correct=%s\n' \
                "$N" "$T" "$R" "$sp" "$eff" "$corr"
        done
    done
done

echo
echo "Done. Wrote $(( $(wc -l < "$OUT") - 1 )) data rows to $OUT"
