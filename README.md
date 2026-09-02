# parallel-matrix-multiply

Reference implementation for the report **"Parallel Matrix Multiplication
Using OpenMP"** (21CSE360T – High Performance Computing).

It multiplies two dense `N x N` matrices with

1. a plain sequential triple-nested loop, and
2. an OpenMP version that distributes the outer `i`-loop across `T` threads
   with `#pragma omp parallel for schedule(static) num_threads(T)`,

then times both with `omp_get_wtime()`, verifies the parallel result against
the sequential one (element-wise, `epsilon = 1e-6`) and reports **speedup**
(`time_seq / time_par`) and **efficiency** (`speedup / T * 100`).

## Project layout

```
parallel-matrix-multiply/
├── src/
│   └── matrix_multiply_openmp.cpp   core program (interactive / --sequential / --parallel / --benchmark)
├── scripts/
│   ├── benchmark.sh                 sweep sizes x threads -> results/benchmark.csv
│   └── plot_results.py              median of runs -> plots/*.png
├── results/
│   └── benchmark.csv                benchmark output (generated)
├── plots/
│   ├── speedup_vs_threads.png       generated
│   └── efficiency_vs_threads.png    generated
├── Makefile
└── README.md
```

## Quick start

```
git clone https://github.com/nk0210/HPC_Matrix_Multiplication.git
cd HPC_Matrix_Multiplication
make            # or: mingw32-make        (build)
./matrix_multiply                          # interactive: enter N, then T
```

Full report pipeline (needs Python + pandas + matplotlib):

```
make benchmark  # -> results/benchmark.csv   (a few minutes)
make plots      # -> plots/*.png
```

## Requirements

| Tool | Notes |
|------|-------|
| `g++` with OpenMP | GCC/MinGW-w64; build uses `-O2 -fopenmp -Wall` |
| GNU `make` | on Windows/MinGW it is installed as `mingw32-make` |
| `bash` | for `scripts/benchmark.sh` (Git Bash / MSYS2 / WSL on Windows) |
| Python 3 + `pandas` + `matplotlib` | for `scripts/plot_results.py` |

### Install the toolchain

**Windows (MSYS2 / MinGW-w64)** – in an "MSYS2 UCRT64" shell:

```
pacman -S --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-make python
```

Then use `mingw32-make` wherever this README says `make`, and run the shell
scripts from the MSYS2 / Git Bash shell.

**Ubuntu / Debian / WSL:**

```
sudo apt update && sudo apt install -y build-essential python3-pip
```

**macOS (Homebrew):**

```
brew install gcc make python
```

(Apple's `clang` has no OpenMP out of the box; use Homebrew `g++-14` or add
`libomp`.)

### Install the Python dependencies

```
python -m pip install pandas matplotlib
```

## Build

```
make            # or: mingw32-make
```

This produces the `matrix_multiply` executable (`matrix_multiply.exe` on
Windows) in the project root. The exact compile command is:

```
g++ -O2 -fopenmp -Wall -o matrix_multiply src/matrix_multiply_openmp.cpp
```

## Run – interactive mode (default)

```
make run        # builds first, then runs ./matrix_multiply
# or directly:
./matrix_multiply
```

It prompts for the matrix size `N` and the thread count `T`, then computes the
product **both ways** and prints a separate, self-contained block for each:

* `======== SEQUENTIAL ========` – 3x3 corner of the result + sequential time.
* `======== PARALLEL (T threads) ========` – 3x3 corner of the result, the
  thread/row distribution, and the parallel time.
* `======== COMPARISON ========` – speedup, efficiency and the `1e-6`
  verification of the two result matrices against each other.

Example session:

```
Enter matrix size N: 1000
Enter number of threads T: 4
...
======== SEQUENTIAL ========
Sequential time (s)  : 3.412007
======== PARALLEL (4 threads) ========
Parallel time (s)    : 0.921334
======== COMPARISON ========
 Speedup (time_seq/time_par)  : 3.7034
 Efficiency (%)               : 92.5850
 Verification (eps=1e-6)      : PASSED
```

## Run – one version at a time

Compute and report only one version (same fixed seed, so the inputs match):

```
./matrix_multiply --sequential N        # e.g. ./matrix_multiply --sequential 1000
./matrix_multiply --parallel N T        # e.g. ./matrix_multiply --parallel 1000 8
```

or via the Makefile (defaults `N=1000`, `T=4`):

```
make seq N=1000
make par N=1000 T=8
```

`--sequential` prints the sequential result block and its time only;
`--parallel` prints the parallel result block, the thread/row distribution and
its time only.

## Run – benchmark mode (scriptable, no stdin)

```
./matrix_multiply --benchmark N T
```

Runs each version once and prints **exactly one CSV line**:

```
N,T,time_seq,time_par,speedup,efficiency,correct
```

for example:

```
$ ./matrix_multiply --benchmark 200 4
200,4,0.010451,0.003128,3.3411,83.5288,1
```

`correct` is `1` when the parallel result matches the sequential one within
`1e-6`, else `0`.

## Reproduce the Section 8 benchmark table

`scripts/benchmark.sh` sweeps the grid used in the report:

* matrix sizes: **500, 1000, 2000**
* thread counts: **1, 2, 4, 8**
* **3 runs** per combination

and writes every run to `results/benchmark.csv` with the header

```
matrix_size,threads,run,time_seq,time_par,speedup,efficiency,correct
```

Run it via the Makefile:

```
make benchmark      # or: mingw32-make benchmark
```

or directly:

```
bash scripts/benchmark.sh
```

The full sweep is 3 sizes x 4 thread counts x 3 runs = 36 rows, i.e. 36
sequential plus 36 parallel multiplications up to `2000 x 2000`, and can take
several minutes (about 8 minutes on a 24-thread machine). For a quick end-to-end check you can shrink the grid
with environment variables (the CSV format is identical):

```
MM_SIZES="200 400" MM_THREADS="1 2 4" MM_RUNS=1 bash scripts/benchmark.sh
```

## Reproduce the Section 8 graphs

`scripts/plot_results.py` reads `results/benchmark.csv`, takes the **median of
the 3 runs** per `(matrix_size, threads)` and writes two figures to `plots/`:

* `speedup_vs_threads.png` – one line per matrix size, plus the ideal
  linear-speedup reference line (`speedup = threads`).
* `efficiency_vs_threads.png` – one line per matrix size.

```
make plots          # or: mingw32-make plots
# or directly:
python scripts/plot_results.py
```

## Full pipeline

```
make            # build
make benchmark  # results/benchmark.csv
make plots      # plots/speedup_vs_threads.png, plots/efficiency_vs_threads.png
```

## Clean

```
make clean
```

Removes the binary, `results/benchmark.csv` and the generated PNGs.

## Implementation notes

* `src/matrix_multiply_openmp.cpp` functions:
  `allocateMatrix`, `freeMatrix`, `randomFill`, `zeroFill`,
  `sequentialMultiply`, `parallelMultiply`, `verifyResults` (1e-6 epsilon),
  `printCorner` (3x3 preview), `printThreadDistribution`.
* Matrices are allocated as an array of row pointers (`double**`).
* Both matrices are filled from a fixed seed (`srand(42)`) so sequential and
  parallel inputs are identical and runs are comparable.
* Only the outer `i`-loop is parallelised; each thread writes a disjoint set
  of rows of `C`, so no synchronisation is needed inside the kernel.
