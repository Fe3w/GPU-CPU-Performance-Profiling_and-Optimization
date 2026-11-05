# -----------------------------------------------------------------------------
# GPU-CPU-Performance-Profiling_and-Optimization — Makefile (W04 targets only)
#
# W01–W03 are built outside this Makefile (their build recipe is not in the
# repo; see docs/W04_PROGRESS.md, open issues). Nothing here touches them.
#
#   make w04          build the W04 self-tests and every W04 stage that exists
#   make w04-test     build, then run tests/test_w04_correctness.sh
#   make clean-w04    remove build/W04_* only
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
W04_SELFTESTS := $(BUILD)/W04_ATTENTION_SELFTEST $(BUILD)/W04_ATTENTION_SELFTEST_GPU
W04_HEADERS  := $(wildcard src/common/*.hpp src/common/*.cuh) $(BUILD)/w04_build_info.h

# ---- per-stage flags (decisions recorded in docs/W04_PROGRESS.md) ----------
# CPU S1: -O2, as in the plan (decision C5); the -O3 / -march=native /
# -ffast-math comparison is the P1 compiler-flag study.
$(BUILD)/W04_ATTENTION_CPU_S1_BASE:          W04_CXXFLAGS := -O2 -std=c++17 -fopenmp -Wall -Wextra
$(BUILD)/W04_ATTENTION_CPU_S3_HWACC_AVX2:    W04_CXXFLAGS += -mavx2 -mfma
$(BUILD)/W04_ATTENTION_GPU_REF_CUBLAS:       W04_NVLDLIBS += -lcublas

.PHONY: help w04 w04-test clean-w04 FORCE
.DEFAULT_GOAL := help

help:
	@echo "targets: w04 | w04-test | clean-w04"
	@echo "  W04 stages found: $(notdir $(W04_CPU_BINS) $(W04_GPU_BINS))"

w04: $(W04_SELFTESTS) $(W04_CPU_BINS) $(W04_GPU_BINS)

w04-test: w04
	bash tests/test_w04_correctness.sh

clean-w04:
	rm -f $(BUILD)/W04_* $(BUILD)/w04_build_info.h

$(BUILD):
	mkdir -p $@

# Git SHA (+ "-dirty" if src/, tests/ or the Makefile have uncommitted or
# untracked changes, i.e. the binary is not reproducible from the SHA). FORCE runs the
# recipe every time, but the file is only rewritten when the content changes,
# so binaries are relinked only when the SHA/dirty state actually changes.
$(BUILD)/w04_build_info.h: FORCE | $(BUILD)
	@sha=$$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown); \
	[ -z "$$(git status --porcelain -- src tests Makefile 2>/dev/null)" ] || sha="$$sha-dirty"; \
	new="#define W04_GIT_SHA \"$$sha\""; \
	if [ ! -f $@ ] || [ "$$(cat $@)" != "$$new" ]; then echo "$$new" > $@; echo "build info: $$sha"; fi

# ---- rules ------------------------------------------------------------------
$(BUILD)/W04_%: src/cpu/W04_ATTENTION/W04_%.cpp $(W04_HEADERS) | $(BUILD)
	$(CXX) $(W04_CXXFLAGS) $(INC) -DW04_COMPILER_FLAGS='"$(CXX) $(W04_CXXFLAGS)"' $< -o $@ $(W04_LDLIBS)

$(BUILD)/W04_%: src/gpu/W04_ATTENTION/W04_%.cu $(W04_HEADERS) | $(BUILD)
	$(NVCC) $(W04_NVCCFLAGS) $(INC) -DW04_COMPILER_FLAGS='"$(NVCC) $(W04_NVCCFLAGS)"' $< -o $@ $(W04_NVLDLIBS)

$(BUILD)/W04_ATTENTION_SELFTEST: tests/w04_selftest.cpp $(W04_HEADERS) | $(BUILD)
	$(CXX) $(W04_CXXFLAGS) $(INC) -DW04_COMPILER_FLAGS='"$(CXX) $(W04_CXXFLAGS)"' $< -o $@ $(W04_LDLIBS)

$(BUILD)/W04_ATTENTION_SELFTEST_GPU: tests/w04_selftest_gpu.cu $(W04_HEADERS) | $(BUILD)
	$(NVCC) $(W04_NVCCFLAGS) $(INC) -DW04_COMPILER_FLAGS='"$(NVCC) $(W04_NVCCFLAGS)"' $< -o $@ $(W04_NVLDLIBS)
