# Makefile - Parallel Matrix Multiplication Using OpenMP
# 21CSE360T - High Performance Computing
#
# Targets:
#   make            build the matrix_multiply binary
#   make run        build, then run the interactive program (computes both)
#   make seq        build, then run sequential only   (make seq N=800)
#   make par        build, then run parallel only     (make par N=800 T=8)
#   make benchmark  build, then run scripts/benchmark.sh (-> results/benchmark.csv)
#   make plots      run scripts/plot_results.py (-> plots/*.png)
#   make clean      remove the binary and generated files
#
# On Windows/MinGW the GNU make executable is usually called "mingw32-make";
# substitute that for "make" in the commands above.
#
# Plain "=" (not "?=") is used below so a stale CXX baked into some MinGW make
# builds is ignored; you can still override on the command line, e.g.
#   make CXX=clang++
CXX      = g++
CXXFLAGS = -O2 -fopenmp -Wall
PYTHON   = python
BASH     = bash

# Defaults for the "seq" / "par" convenience targets; override on the command
# line, e.g.  make par N=1000 T=8
N := 1000
T := 4

SRC := src/matrix_multiply_openmp.cpp
BIN := matrix_multiply

.PHONY: all run seq par benchmark plots clean

all: $(BIN)

$(BIN): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(BIN) $(SRC)

run: $(BIN)
	./$(BIN)

seq: $(BIN)
	./$(BIN) --sequential $(N)

par: $(BIN)
	./$(BIN) --parallel $(N) $(T)

benchmark: $(BIN)
	$(BASH) scripts/benchmark.sh

plots:
	$(PYTHON) scripts/plot_results.py

clean:
	$(RM) $(BIN) $(BIN).exe
	$(RM) results/benchmark.csv
	$(RM) plots/speedup_vs_threads.png plots/efficiency_vs_threads.png
