# -----------------------------------------------------------------------------
# GPU-CPU-Performance-Profiling_and-Optimization — Makefile
#
#   make w01 | w02 | w03 | legacy   W01–W03 stage binaries (legacy = all three)
#   make w04          build the W04 self-tests and every W04 stage that exists
#   make w04-test     build, then run tests/test_w04_correctness.sh
#   make w04-flagstudy  P1 compiler-flag study binaries in build/flagstudy/
#   make clean-w04    remove build/W04_* and build/flagstudy/ only
#
# Stages are found by wildcard, so a new src/{cpu,gpu}/W04_ATTENTION/<name>.{cpp,cu}
# is built as build/<name> without editing this file. Per-stage flag changes
# are target-specific variables below. The exact flags used are compiled into
# each binary (W04_COMPILER_FLAGS) and written to every CSV row.
# -----------------------------------------------------------------------------
SHELL := /bin/bash

CXX   := g++
NVCC  := nvcc
BUILD := build
INC   := -Isrc/common -I$(BUILD)

W04_CXXFLAGS  := -O3 -std=c++17 -fopenmp -Wall -Wextra
W04_NVCCFLAGS := -O3 -std=c++17 -lineinfo -gencode arch=compute_89,code=sm_89 \
                 -Xcompiler -fopenmp -Xcompiler -Wall
W04_LDLIBS    :=
W04_NVLDLIBS  := -lgomp

W04_CPU_SRCS := $(wildcard src/cpu/W04_ATTENTION/*.cpp)
W04_GPU_SRCS := $(wildcard src/gpu/W04_ATTENTION/*.cu)
W04_CPU_BINS := $(patsubst src/cpu/W04_ATTENTION/%.cpp,$(BUILD)/%,$(W04_CPU_SRCS))
W04_GPU_BINS := $(patsubst src/gpu/W04_ATTENTION/%.cu,$(BUILD)/%,$(W04_GPU_SRCS))
W04_SELFTESTS := $(BUILD)/W04_ATTENTION_SELFTEST $(BUILD)/W04_ATTENTION_SELFTEST_GPU $(BUILD)/W04_EXP_AVX2_TEST
W04_HEADERS  := $(wildcard src/common/*.hpp src/common/*.cuh src/gpu/W04_ATTENTION/*.cuh) $(BUILD)/w04_build_info.h

# ---- per-stage flags (decisions recorded in docs/W04_PROGRESS.md) ----------
# CPU S1: -O2, as in the plan (decision C5); the -O3 / -march=native /
# -ffast-math comparison is the P1 compiler-flag study.
$(BUILD)/W04_ATTENTION_CPU_S1_BASE:          W04_CXXFLAGS := -O2 -std=c++17 -fopenmp -Wall -Wextra
$(BUILD)/W04_ATTENTION_CPU_S3_HWACC_AVX2:    W04_CXXFLAGS += -mavx2 -mfma
$(BUILD)/W04_ATTENTION_GPU_REF_CUBLAS:       W04_NVLDLIBS += -lcublas

.PHONY: help w01 w02 w03 legacy w04 w04-test w04-flagstudy clean-w04 FORCE
.DEFAULT_GOAL := help

help:
	@echo "targets: w01 | w02 | w03 | legacy | w04 | w04-test | w04-flagstudy | clean-w04"
	@echo "  W04 stages found: $(notdir $(W04_CPU_BINS) $(W04_GPU_BINS))"

w04: $(W04_SELFTESTS) $(W04_CPU_BINS) $(W04_GPU_BINS)

w04-test: w04
	bash tests/test_w04_correctness.sh

clean-w04:
	rm -f $(BUILD)/W04_* $(BUILD)/w04_build_info.h
	rm -rf $(BUILD)/flagstudy

$(BUILD):
	mkdir -p $@

# Git SHA (+ "-dirty" if src/, tests/ or the Makefile have uncommitted or
# untracked changes, i.e. the binary is not reproducible from the SHA).
# "nogit" if there is no git binary, no repository (e.g. a source archive) or
# no commit yet; the self-test accepts it with a warning. FORCE runs the
# recipe every time, but the file is only rewritten when the content changes,
# so binaries are relinked only when the SHA/dirty state actually changes.
$(BUILD)/w04_build_info.h: FORCE | $(BUILD)
	@if command -v git >/dev/null 2>&1 && sha=$$(git rev-parse --short=12 HEAD 2>/dev/null); then \
	    [ -z "$$(git status --porcelain -- src tests Makefile 2>/dev/null)" ] || sha="$$sha-dirty"; \
	else sha=nogit; fi; \
	new="#define W04_GIT_SHA \"$$sha\""; \
	if [ ! -f $@ ] || [ "$$(cat $@)" != "$$new" ]; then echo "$$new" > $@; echo "build info: $$sha"; fi

# ---- rules ------------------------------------------------------------------
$(BUILD)/W04_%: src/cpu/W04_ATTENTION/W04_%.cpp $(W04_HEADERS) | $(BUILD)
	$(CXX) $(W04_CXXFLAGS) $(INC) -DW04_COMPILER_FLAGS='"$(CXX) $(W04_CXXFLAGS)"' $< -o $@ $(W04_LDLIBS)

$(BUILD)/W04_%: src/gpu/W04_ATTENTION/W04_%.cu $(W04_HEADERS) | $(BUILD)
	$(NVCC) $(W04_NVCCFLAGS) $(INC) -DW04_COMPILER_FLAGS='"$(NVCC) $(W04_NVCCFLAGS)"' $< -o $@ $(W04_NVLDLIBS)

$(BUILD)/W04_ATTENTION_SELFTEST: tests/w04_selftest.cpp $(W04_HEADERS) | $(BUILD)
	$(CXX) $(W04_CXXFLAGS) $(INC) -DW04_COMPILER_FLAGS='"$(CXX) $(W04_CXXFLAGS)"' $< -o $@ $(W04_LDLIBS)

$(BUILD)/W04_EXP_AVX2_TEST: tests/w04_exp_avx2_test.cpp $(W04_HEADERS) | $(BUILD)
	$(CXX) $(W04_CXXFLAGS) -mavx2 -mfma $(INC) $< -o $@ $(W04_LDLIBS)

$(BUILD)/W04_ATTENTION_SELFTEST_GPU: tests/w04_selftest_gpu.cu $(W04_HEADERS) | $(BUILD)
	$(NVCC) $(W04_NVCCFLAGS) $(INC) -DW04_COMPILER_FLAGS='"$(NVCC) $(W04_NVCCFLAGS)"' $< -o $@ $(W04_NVLDLIBS)

# ---- P1 compiler-flag study -------------------------------------------------
# The S2 source (portable C++) is the auto-vectorized subject, the S3 source
# (intrinsics) the hand-vectorized one; S3 always needs -mavx2 -mfma. Each
# binary is build/flagstudy/<stage>.<tag>; the flags are compiled in and land
# in the CSV compiler_flags column. The self-test is also built with the most
# aggressive set, because the header-only double reference is compiled with
# the same flags as the stage it checks. S2 builds also write the compiler's
# vectorization report (<binary>.optinfo).
FS_DIR      := $(BUILD)/flagstudy
FS_TAGS     := O2 O3n O3f O3nf
FS_FLAGS_O2   := -O2
FS_FLAGS_O3n  := -O3 -march=native
FS_FLAGS_O3f  := -O3 -ffast-math
FS_FLAGS_O3nf := -O3 -march=native -ffast-math
FS_BASE     := -std=c++17 -fopenmp -Wall -Wextra
FS_S2       := W04_ATTENTION_CPU_S2_SWOPT_BLOCKED_OMP
FS_S3       := W04_ATTENTION_CPU_S3_HWACC_AVX2
FS_BINS     := $(foreach t,$(FS_TAGS),$(FS_DIR)/$(FS_S2).$(t) $(FS_DIR)/$(FS_S3).$(t)) \
               $(FS_DIR)/W04_ATTENTION_SELFTEST.O3nf

w04-flagstudy: $(FS_BINS)

$(FS_DIR):
	mkdir -p $@

define FS_RULES
$(FS_DIR)/$(FS_S2).$(1): src/cpu/W04_ATTENTION/$(FS_S2).cpp $(W04_HEADERS) | $(FS_DIR)
	$(CXX) $(FS_FLAGS_$(1)) $(FS_BASE) -fopt-info-vec-optimized=$$@.optinfo $(INC) -DW04_COMPILER_FLAGS='"$(CXX) $(FS_FLAGS_$(1)) $(FS_BASE)"' $$< -o $$@ $(W04_LDLIBS)
$(FS_DIR)/$(FS_S3).$(1): src/cpu/W04_ATTENTION/$(FS_S3).cpp $(W04_HEADERS) | $(FS_DIR)
	$(CXX) $(FS_FLAGS_$(1)) $(FS_BASE) -mavx2 -mfma $(INC) -DW04_COMPILER_FLAGS='"$(CXX) $(FS_FLAGS_$(1)) $(FS_BASE) -mavx2 -mfma"' $$< -o $$@ $(W04_LDLIBS)
endef
$(foreach t,$(FS_TAGS),$(eval $(call FS_RULES,$(t))))

$(FS_DIR)/W04_ATTENTION_SELFTEST.O3nf: tests/w04_selftest.cpp $(W04_HEADERS) | $(FS_DIR)
	$(CXX) $(FS_FLAGS_O3nf) $(FS_BASE) $(INC) -DW04_COMPILER_FLAGS='"$(CXX) $(FS_FLAGS_O3nf) $(FS_BASE)"' $< -o $@ $(W04_LDLIBS)

# ---- W01–W03 (legacy workloads) ---------------------------------------------
# One binary per stage source, build/<source name>. Each source writes its own
# compiler flags into its CSV as a fixed string; the flags below are exactly
# those strings, so the CSV column stays true. Exception: W03 CPU S2/S3 add
# -DGEMM_BLOCK_SIZE=128, the tile used for the recorded W03 results (B128),
# which their flag string does not mention. Superseded *_v1.cpp files and the
# non-stage files in src/cpu/ are not built.
L_CPU_O0     := -O0 -std=c++17 -g
L_CPU_O3     := -O3 -fopenmp -fno-tree-vectorize -fno-tree-slp-vectorize -std=c++17 -g
L_GPU        := -O3 -std=c++17 -g -gencode arch=compute_89,code=sm_89
W01_BINS := $(addprefix $(BUILD)/W01_VECTOR_ADD_,CPU_S1_BASE CPU_S2_SWOPT_OMP CPU_S3_HWACC_AVX2 \
                GPU_S1_BASE_NAIVE GPU_S2_SWOPT_COALESCED)
W02_BINS := $(addprefix $(BUILD)/W02_DOT_PRODUCT_,CPU_S1_BASE CPU_S2_SWOPT_OMP CPU_S3_HWACC_AVX2_FMA \
                GPU_S1_BASE_NAIVE GPU_S2_SWOPT_BLOCK_REDUCE)
W03_BINS := $(addprefix $(BUILD)/W03_GEMM_,CPU_S1_BASE CPU_S2_SWOPT_CACHE_BLOCK CPU_S3_HWACC_AVX2_FMA \
                GPU_S1_BASE_NAIVE GPU_S2_SWOPT_SHARED_TILE GPU_S3_HWACC_TENSOR_CORE)

w01: $(W01_BINS)
w02: $(W02_BINS)
w03: $(W03_BINS)
legacy: w01 w02 w03

$(BUILD)/W01_VECTOR_ADD_CPU_S1_BASE:          L_FLAGS := -O0 -std=c++17 -g
$(BUILD)/W01_VECTOR_ADD_CPU_S2_SWOPT_OMP:     L_FLAGS := -O0 -std=c++17 -g -fopenmp
$(BUILD)/W01_VECTOR_ADD_CPU_S3_HWACC_AVX2:    L_FLAGS := -O0 -std=c++17 -g -fopenmp -mavx2
$(BUILD)/W02_DOT_PRODUCT_CPU_S1_BASE:         L_FLAGS := $(L_CPU_O0)
$(BUILD)/W02_DOT_PRODUCT_CPU_S2_SWOPT_OMP:    L_FLAGS := $(L_CPU_O3)
$(BUILD)/W02_DOT_PRODUCT_CPU_S3_HWACC_AVX2_FMA: L_FLAGS := -O3 -mavx2 -mfma -fopenmp -fno-tree-vectorize -fno-tree-slp-vectorize -std=c++17 -g
$(BUILD)/W03_GEMM_CPU_S1_BASE:                L_FLAGS := $(L_CPU_O0)
$(BUILD)/W03_GEMM_CPU_S2_SWOPT_CACHE_BLOCK:   L_FLAGS := $(L_CPU_O3) -DGEMM_BLOCK_SIZE=128
$(BUILD)/W03_GEMM_CPU_S3_HWACC_AVX2_FMA:      L_FLAGS := -O3 -fopenmp -mavx2 -mfma -fno-tree-vectorize -fno-tree-slp-vectorize -std=c++17 -g -DGEMM_BLOCK_SIZE=128
$(BUILD)/W01_VECTOR_ADD_GPU_S1_BASE_NAIVE:    L_FLAGS := -O3 -std=c++17 -lineinfo
$(BUILD)/W01_VECTOR_ADD_GPU_S2_SWOPT_COALESCED: L_FLAGS := -O3 -std=c++17 -lineinfo -gencode arch=compute_89,code=sm_89
$(BUILD)/W02_DOT_PRODUCT_GPU_%:               L_FLAGS := $(L_GPU)
$(BUILD)/W03_GEMM_GPU_%:                      L_FLAGS := $(L_GPU)

$(BUILD)/W01_%: src/cpu/W01_VECTOR_ADD/W01_%.cpp | $(BUILD)
	$(CXX) $(L_FLAGS) $< -o $@
$(BUILD)/W02_%: src/cpu/W02_DOT_PRODUCT/W02_%.cpp | $(BUILD)
	$(CXX) $(L_FLAGS) $< -o $@
$(BUILD)/W03_%: src/cpu/W03_GEMM/W03_%.cpp | $(BUILD)
	$(CXX) $(L_FLAGS) $< -o $@
$(BUILD)/W01_%: src/gpu/W01_VECTOR_ADD/W01_%.cu | $(BUILD)
	$(NVCC) $(L_FLAGS) $< -o $@
$(BUILD)/W02_%: src/gpu/W02_DOT_PRODUCT/W02_%.cu | $(BUILD)
	$(NVCC) $(L_FLAGS) $< -o $@
$(BUILD)/W03_%: src/gpu/W03_GEMM/W03_%.cu | $(BUILD)
	$(NVCC) $(L_FLAGS) $< -o $@
