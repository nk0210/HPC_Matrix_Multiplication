// matrix_multiply_openmp.cpp
//
// Parallel Matrix Multiplication Using OpenMP
// 21CSE360T - High Performance Computing
//
// Execution modes:
//
//   1. Interactive (default, no arguments):
//        ./matrix_multiply
//      Prompts for the matrix size N and the thread count T, then computes
//      the product BOTH ways - sequential first, parallel second - printing
//      a separate, self-contained result block for each, followed by a
//      comparison block (verification, speedup, efficiency).
//
//   2. Sequential only:
//        ./matrix_multiply --sequential N
//      Computes C = A * B with the plain triple loop and reports only the
//      sequential result and its time.
//
//   3. Parallel only:
//        ./matrix_multiply --parallel N T
//      Computes C = A * B with OpenMP on T threads and reports only the
//      parallel result, the thread/row distribution and its time.
//
//   4. Benchmark (scriptable, no stdin needed):
//        ./matrix_multiply --benchmark N T
//      Runs both versions once and prints exactly one CSV line:
//        N,T,time_seq,time_par,speedup,efficiency,correct
//
// Build:
//   g++ -O2 -fopenmp -Wall -o matrix_multiply src/matrix_multiply_openmp.cpp

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <omp.h>

// ---------------------------------------------------------------------------
// Memory management
// ---------------------------------------------------------------------------

// Allocate an n x n matrix as an array of row pointers.
double** allocateMatrix(int n) {
    double** mat = new double*[n];
    for (int i = 0; i < n; ++i) {
        mat[i] = new double[n];
    }
    return mat;
}

// Free a matrix previously returned by allocateMatrix.
void freeMatrix(double** mat, int n) {
    if (mat == nullptr) return;
    for (int i = 0; i < n; ++i) {
        delete[] mat[i];
    }
    delete[] mat;
}

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------

// Fill the matrix with pseudo-random values in [0, 10).
void randomFill(double** mat, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            mat[i][j] = (static_cast<double>(rand()) / RAND_MAX) * 10.0;
        }
    }
}

// Set every element of the matrix to zero.
void zeroFill(double** mat, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            mat[i][j] = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Multiplication kernels
// ---------------------------------------------------------------------------

// Classic triple-nested-loop sequential multiplication: C = A * B.
void sequentialMultiply(double** A, double** B, double** C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

// OpenMP parallel multiplication: C = A * B.
// The outer i-loop is distributed across `threads` threads with a static
// schedule, so each thread owns a contiguous block of rows.
void parallelMultiply(double** A, double** B, double** C, int n, int threads) {
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Verification and reporting
// ---------------------------------------------------------------------------

// Return true when every pair of elements agrees within 1e-6.
bool verifyResults(double** C1, double** C2, int n) {
    const double epsilon = 1e-6;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (std::fabs(C1[i][j] - C2[i][j]) > epsilon) {
                return false;
            }
        }
    }
    return true;
}

// Print the top-left 3x3 corner of a matrix as a quick sanity preview.
void printCorner(double** mat, int n, const std::string& name) {
    int m = (n < 3) ? n : 3;
    std::cout << "Top-left " << m << "x" << m << " corner of " << name << ":\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            std::cout << std::setw(12) << std::fixed << std::setprecision(4)
                      << mat[i][j];
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// Show how the static schedule splits the n rows across the threads.
void printThreadDistribution(int n, int threads) {
    std::cout << "Thread / row distribution for schedule(static), "
              << threads << " thread(s) over " << n << " rows:\n";

    std::vector<int> rowsPerThread(threads, 0);
    std::vector<int> firstRow(threads, -1);
    std::vector<int> lastRow(threads, -1);

    #pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            if (firstRow[tid] == -1) {
                firstRow[tid] = i;
            }
            lastRow[tid] = i;
            rowsPerThread[tid]++;
        }
    }

    for (int t = 0; t < threads; ++t) {
        std::cout << "  Thread " << std::setw(2) << t << " -> ";
        if (rowsPerThread[t] == 0) {
            std::cout << "(no rows)\n";
        } else {
            std::cout << "rows [" << firstRow[t] << " .. " << lastRow[t]
                      << "], " << rowsPerThread[t] << " row(s)\n";
        }
    }
    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// Per-version runners (each computes ONE version and reports it on its own)
// ---------------------------------------------------------------------------

// Compute C = A * B sequentially, print its own result block, return the time.
double runSequentialOnly(double** A, double** B, double** C, int n, bool preview) {
    zeroFill(C, n);
    double t0 = omp_get_wtime();
    sequentialMultiply(A, B, C, n);
    double t1 = omp_get_wtime();
    double dt = t1 - t0;

    std::cout << "======== SEQUENTIAL ========\n";
    if (preview) {
        printCorner(C, n, "C = A * B (sequential)");
    }
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Matrix size          : " << n << " x " << n << "\n";
    std::cout << "Sequential time (s)  : " << dt << "\n\n";
    return dt;
}

// Compute C = A * B in parallel, print its own result block, return the time.
double runParallelOnly(double** A, double** B, double** C, int n, int threads,
                       bool preview) {
    zeroFill(C, n);
    double t0 = omp_get_wtime();
    parallelMultiply(A, B, C, n, threads);
    double t1 = omp_get_wtime();
    double dt = t1 - t0;

    std::cout << "======== PARALLEL (" << threads << " threads) ========\n";
    if (preview) {
        printCorner(C, n, "C = A * B (parallel)");
    }
    printThreadDistribution(n, threads);
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Matrix size          : " << n << " x " << n << "\n";
    std::cout << "Threads              : " << threads << "\n";
    std::cout << "Parallel time (s)    : " << dt << "\n\n";
    return dt;
}

// ---------------------------------------------------------------------------
// Drivers
// ---------------------------------------------------------------------------

// Mode: --sequential N
int runSequentialMode(int n) {
    srand(42);
    double** A = allocateMatrix(n);
    double** B = allocateMatrix(n);
    double** C = allocateMatrix(n);
    randomFill(A, n);
    randomFill(B, n);

    printCorner(A, n, "A");
    printCorner(B, n, "B");
    runSequentialOnly(A, B, C, n, true);

    freeMatrix(A, n);
    freeMatrix(B, n);
    freeMatrix(C, n);
    return 0;
}

// Mode: --parallel N T
int runParallelMode(int n, int threads) {
    srand(42);
    double** A = allocateMatrix(n);
    double** B = allocateMatrix(n);
    double** C = allocateMatrix(n);
    randomFill(A, n);
    randomFill(B, n);

    printCorner(A, n, "A");
    printCorner(B, n, "B");
    runParallelOnly(A, B, C, n, threads, true);

    freeMatrix(A, n);
    freeMatrix(B, n);
    freeMatrix(C, n);
    return 0;
}

// Mode: --benchmark N T
// Runs both versions and emits a single CSV row:
//   N,T,time_seq,time_par,speedup,efficiency,correct
int runBenchmark(int n, int threads) {
    srand(42);

    double** A = allocateMatrix(n);
    double** B = allocateMatrix(n);
    double** Cseq = allocateMatrix(n);
    double** Cpar = allocateMatrix(n);

    randomFill(A, n);
    randomFill(B, n);
    zeroFill(Cseq, n);
    zeroFill(Cpar, n);

    double t0 = omp_get_wtime();
    sequentialMultiply(A, B, Cseq, n);
    double t1 = omp_get_wtime();
    double time_seq = t1 - t0;

    t0 = omp_get_wtime();
    parallelMultiply(A, B, Cpar, n, threads);
    t1 = omp_get_wtime();
    double time_par = t1 - t0;

    bool correct = verifyResults(Cseq, Cpar, n);
    double speedup = (time_par > 0.0) ? (time_seq / time_par) : 0.0;
    double efficiency = (speedup / threads) * 100.0;

    std::cout << n << ','
              << threads << ','
              << std::fixed << std::setprecision(6) << time_seq << ','
              << std::setprecision(6) << time_par << ','
              << std::setprecision(4) << speedup << ','
              << std::setprecision(4) << efficiency << ','
              << (correct ? 1 : 0) << '\n';

    freeMatrix(A, n);
    freeMatrix(B, n);
    freeMatrix(Cseq, n);
    freeMatrix(Cpar, n);

    return correct ? 0 : 1;
}

// Mode: interactive (default). Computes BOTH versions separately, prints a
// self-contained block for each, then a comparison block.
int runInteractive() {
    std::cout << "=============================================\n";
    std::cout << " Parallel Matrix Multiplication Using OpenMP\n";
    std::cout << " 21CSE360T - High Performance Computing\n";
    std::cout << "=============================================\n\n";
    std::cout << "Maximum threads available on this machine: "
              << omp_get_max_threads() << "\n\n";

    int n = 0;
    int threads = 0;

    std::cout << "Enter matrix size N: ";
    if (!(std::cin >> n) || n <= 0) {
        std::cerr << "Invalid matrix size.\n";
        return 1;
    }

    std::cout << "Enter number of threads T: ";
    if (!(std::cin >> threads) || threads <= 0) {
        std::cerr << "Invalid thread count.\n";
        return 1;
    }
    std::cout << "\n";

    srand(42);

    double** A = allocateMatrix(n);
    double** B = allocateMatrix(n);
    double** Cseq = allocateMatrix(n);
    double** Cpar = allocateMatrix(n);

    randomFill(A, n);
    randomFill(B, n);

    printCorner(A, n, "A");
    printCorner(B, n, "B");

    // ---- sequential, computed and reported on its own ----
    double time_seq = runSequentialOnly(A, B, Cseq, n, true);

    // ---- parallel, computed and reported on its own ----
    double time_par = runParallelOnly(A, B, Cpar, n, threads, true);

    // ---- comparison of the two ----
    bool correct = verifyResults(Cseq, Cpar, n);
    double speedup = (time_par > 0.0) ? (time_seq / time_par) : 0.0;
    double efficiency = (speedup / threads) * 100.0;

    std::cout << "======== COMPARISON ========\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << std::left << std::setw(30) << " Sequential time (s)"
              << ": " << time_seq << "\n";
    std::cout << std::left << std::setw(30) << " Parallel time (s)"
              << ": " << time_par << "\n";
    std::cout << std::setprecision(4);
    std::cout << std::left << std::setw(30) << " Speedup (time_seq/time_par)"
              << ": " << speedup << "\n";
    std::cout << std::left << std::setw(30) << " Efficiency (%)"
              << ": " << efficiency << "\n";
    std::cout << std::left << std::setw(30) << " Verification (eps=1e-6)"
              << ": " << (correct ? "PASSED" : "FAILED") << "\n";
    std::cout << "============================\n";

    freeMatrix(A, n);
    freeMatrix(B, n);
    freeMatrix(Cseq, n);
    freeMatrix(Cpar, n);

    return correct ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc >= 2) {
        std::string mode = argv[1];

        if (mode == "--benchmark") {
            if (argc < 4) {
                std::cerr << "Usage: " << argv[0] << " --benchmark N T\n";
                return 1;
            }
            int n = std::atoi(argv[2]);
            int threads = std::atoi(argv[3]);
            if (n <= 0 || threads <= 0) {
                std::cerr << "N and T must be positive integers.\n";
                return 1;
            }
            return runBenchmark(n, threads);
        }

        if (mode == "--sequential" || mode == "--seq") {
            if (argc < 3) {
                std::cerr << "Usage: " << argv[0] << " --sequential N\n";
                return 1;
            }
            int n = std::atoi(argv[2]);
            if (n <= 0) {
                std::cerr << "N must be a positive integer.\n";
                return 1;
            }
            return runSequentialMode(n);
        }

        if (mode == "--parallel" || mode == "--par") {
            if (argc < 4) {
                std::cerr << "Usage: " << argv[0] << " --parallel N T\n";
                return 1;
            }
            int n = std::atoi(argv[2]);
            int threads = std::atoi(argv[3]);
            if (n <= 0 || threads <= 0) {
                std::cerr << "N and T must be positive integers.\n";
                return 1;
            }
            return runParallelMode(n, threads);
        }

        std::cerr << "Unknown option: " << mode << "\n";
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << "                 (interactive, runs both)\n"
                  << "  " << argv[0] << " --sequential N\n"
                  << "  " << argv[0] << " --parallel N T\n"
                  << "  " << argv[0] << " --benchmark N T\n";
        return 1;
    }

    return runInteractive();
}
