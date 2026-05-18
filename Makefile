# DESCRIPTION
# - Makefile for CARTS
# - Dependencies: ARTS, LLVM, Polygeist
# - Required: CMake, Ninja; LLVM bootstrap: Clang or GCC (see LLVM_C_COMPILER)

# Build verbosity control (set VERBOSE=1 to enable verbose output)
VERBOSE ?= 0
NINJA_FLAGS := $(if $(filter 1,$(VERBOSE)),-v,) $(if $(CARTS_BUILD_JOBS),-j$(CARTS_BUILD_JOBS),)
CMAKE_BUILD_FLAGS := $(if $(filter 1,$(VERBOSE)),--verbose,) $(if $(CARTS_BUILD_JOBS),--parallel $(CARTS_BUILD_JOBS),)

# CMake executable (override via `make CMAKE=/path/to/cmake` or from build.py)
CMAKE ?= cmake
CMAKE_CMD := $(CMAKE)

# Source Directories
CARTS_DIR = ${shell pwd}
ARTS_DIR ?= ${CARTS_DIR}/external/arts
POLYGEIST_DIR ?= ${CARTS_DIR}/external/Polygeist
LLVM_DIR ?= ${POLYGEIST_DIR}/llvm-project

# Output Directories
#
# By default CARTS keeps build/install artifacts in the project checkout for a
# self-contained developer tree. Set CARTS_HOME to move every heavy generated
# artifact to another filesystem while preserving the same source tree.
ifeq ($(strip $(CARTS_HOME)),)
CARTS_OUTPUT_HOME := ${CARTS_DIR}
CARTS_BUILD_DIR ?= build
ARTS_BUILD_DIR ?= $(ARTS_DIR)/build
POLYGEIST_BUILD_DIR ?= $(POLYGEIST_DIR)/build
LLVM_BUILD_DIR ?= $(LLVM_DIR)/build
else
CARTS_OUTPUT_HOME := $(abspath $(CARTS_HOME))
CARTS_BUILD_DIR ?= $(CARTS_OUTPUT_HOME)/build/carts
ARTS_BUILD_DIR ?= $(CARTS_OUTPUT_HOME)/build/arts
POLYGEIST_BUILD_DIR ?= $(CARTS_OUTPUT_HOME)/build/polygeist
LLVM_BUILD_DIR ?= $(CARTS_OUTPUT_HOME)/build/llvm-project
CARTS_TMP_DIR ?= $(CARTS_OUTPUT_HOME)/tmp
export TMPDIR ?= $(CARTS_TMP_DIR)
export TMP ?= $(TMPDIR)
export TEMP ?= $(TMPDIR)
endif

# Install Directories
INSTALL_DIR ?= $(CARTS_OUTPUT_HOME)/.install
CARTS_INSTALL_DIR ?= ${INSTALL_DIR}/carts
ARTS_INSTALL_DIR ?= $(INSTALL_DIR)/arts
LLVM_INSTALL_DIR ?= $(INSTALL_DIR)/llvm
POLYGEIST_INSTALL_DIR ?= $(INSTALL_DIR)/polygeist

LIT_SOURCE_DIR := $(LLVM_DIR)/llvm/utils/lit
LIT_INSTALL_DIR := $(LLVM_INSTALL_DIR)/utils/lit

# Detect platform-appropriate linker (ld64.lld on macOS, ld.lld on Linux)
CARTS_LINKER_PATH ?= $(if $(filter Darwin,$(shell uname)),${LLVM_INSTALL_DIR}/bin/ld64.lld,${LLVM_INSTALL_DIR}/bin/ld.lld)
LLVM_C_COMPILER ?= clang
LLVM_CXX_COMPILER ?= clang++

# When set (from build.py), points the bootstrap Clang at a GCC install via
# --gcc-toolchain
LLVM_GCC_INSTALL_PREFIX ?=
# Non-empty when bootstrap C compiler basename looks like clang / clang-N.
LLVM_BOOTSTRAP_IS_CLANG := $(filter clang%,$(notdir $(LLVM_C_COMPILER)))
# Use --gcc-toolchain for the llvm target only when we have a prefix and Clang bootstrap.
LLVM_LLVM_USE_GCC_TOOLCHAIN := $(and $(LLVM_GCC_INSTALL_PREFIX),$(LLVM_BOOTSTRAP_IS_CLANG))

# Use our installed linker for all downstream cmake builds.
LLVM_RUNTIME_CMAKE_FLAGS = \
	-DCMAKE_EXE_LINKER_FLAGS="--ld-path=$(CARTS_LINKER_PATH)" \
	-DCMAKE_SHARED_LINKER_FLAGS="--ld-path=$(CARTS_LINKER_PATH)" \
	-DCMAKE_MODULE_LINKER_FLAGS="--ld-path=$(CARTS_LINKER_PATH)" \
	$(if $(LLVM_GCC_INSTALL_PREFIX),\
	-DCMAKE_CXX_FLAGS="--gcc-toolchain=$(LLVM_GCC_INSTALL_PREFIX)" \
	-DCMAKE_C_FLAGS="--gcc-toolchain=$(LLVM_GCC_INSTALL_PREFIX)" \
	)

# Shared Docker workspaces can carry stale CMake caches from older branches
# or build systems. Recreate the build directory before reconfiguring when
# the cached generator is not Ninja.
ensure_ninja_build_dir = if [ -f "$(1)/CMakeCache.txt" ] && ! grep -q '^CMAKE_GENERATOR:INTERNAL=Ninja$$' "$(1)/CMakeCache.txt"; then echo "Removing stale CMake build directory $(1) (generator mismatch)..."; rm -rf "$(1)"; fi
ensure_output_dirs = mkdir -p "$(CARTS_OUTPUT_HOME)" $(if $(CARTS_TMP_DIR),"$(CARTS_TMP_DIR)",)

# Targets
all: install

# Polygeist
polygeist-download:
	@if [ ! -d "$(POLYGEIST_DIR)/.git" ]; then \
		echo "Initializing Polygeist submodule..."; \
		git submodule update --init --recursive external/Polygeist; \
	else \
		echo "Polygeist submodule already initialized."; \
	fi 
polygeist:
	@$(call ensure_ninja_build_dir,$(POLYGEIST_BUILD_DIR))
	echo "Building Polygeist..."; \
	$(call ensure_output_dirs); \
	mkdir -p $(POLYGEIST_BUILD_DIR); \
	mkdir -p $(POLYGEIST_INSTALL_DIR); \
	$(CMAKE_CMD) -B $(POLYGEIST_BUILD_DIR) \
		-S $(POLYGEIST_DIR) -G Ninja \
		-DCMAKE_INSTALL_PREFIX=$(POLYGEIST_INSTALL_DIR) \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_C_COMPILER=$(LLVM_INSTALL_DIR)/bin/clang \
		-DCMAKE_CXX_COMPILER=$(LLVM_INSTALL_DIR)/bin/clang++ \
		-DMLIR_DIR=$(LLVM_BUILD_DIR)/lib/cmake/mlir \
		-DClang_DIR=$(LLVM_BUILD_DIR)/lib/cmake/clang \
		-DLLVM_EXTERNAL_LIT="$(LLVM_BUILD_DIR)/bin/llvm-lit" \
		$(LLVM_RUNTIME_CMAKE_FLAGS) \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON;
	ninja $(NINJA_FLAGS) -C $(POLYGEIST_BUILD_DIR) install;
polygeist-clean:
	rm -f -r $(POLYGEIST_BUILD_DIR)
	rm -f -r $(POLYGEIST_INSTALL_DIR)

# LLVM
llvm:
	@set -e; \
	$(call ensure_ninja_build_dir,$(LLVM_BUILD_DIR)); \
	echo "Building LLVM..."; \
	$(call ensure_output_dirs); \
	mkdir -p $(LLVM_BUILD_DIR); \
	mkdir -p $(LLVM_INSTALL_DIR); \
	$(CMAKE_CMD) -B $(LLVM_BUILD_DIR) \
		-S $(LLVM_DIR)/llvm -G Ninja \
		-DCMAKE_INSTALL_PREFIX=$(LLVM_INSTALL_DIR) \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_C_COMPILER=$(LLVM_C_COMPILER) \
		-DCMAKE_CXX_COMPILER=$(LLVM_CXX_COMPILER) \
		-DLLVM_ENABLE_PROJECTS='mlir;clang;lld' \
		-DLLVM_ENABLE_RUNTIMES='libcxx;libcxxabi;libunwind;compiler-rt;openmp' \
		-DLLVM_TARGETS_TO_BUILD='host' \
		-DLLVM_OPTIMIZED_TABLEGEN=ON \
		-DLLVM_ENABLE_ASSERTIONS=ON \
		-DLLVM_BUILD_TOOLS=ON \
		-DLLVM_INCLUDE_TOOLS=ON \
		-DLLVM_INCLUDE_EXAMPLES=OFF \
		-DLLVM_INCLUDE_TESTS=OFF \
		-DLLVM_BUILD_TESTS=OFF \
		-DLLVM_INSTALL_UTILS=ON \
		-DLLVM_INCLUDE_BENCHMARKS=OFF \
		-DLLVM_INCLUDE_UTILS=ON \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		$(if $(LLVM_LLVM_USE_GCC_TOOLCHAIN),\
		-DCMAKE_CXX_FLAGS="--gcc-toolchain=$(LLVM_GCC_INSTALL_PREFIX)" \
		-DCMAKE_C_FLAGS="--gcc-toolchain=$(LLVM_GCC_INSTALL_PREFIX)" \
		-DCROSS_TOOLCHAIN_FLAGS_NATIVE="-DCMAKE_C_COMPILER=$(LLVM_C_COMPILER);-DCMAKE_CXX_COMPILER=$(LLVM_CXX_COMPILER);-DCMAKE_CXX_FLAGS=--gcc-toolchain=$(LLVM_GCC_INSTALL_PREFIX);-DCMAKE_C_FLAGS=--gcc-toolchain=$(LLVM_GCC_INSTALL_PREFIX)" \
		); \
	ninja $(NINJA_FLAGS) -C $(LLVM_BUILD_DIR) install; \
	if [ -f "$(LLVM_BUILD_DIR)/bin/llvm-lit" ]; then \
		mkdir -p $(LLVM_INSTALL_DIR)/bin; \
		cp $(LLVM_BUILD_DIR)/bin/llvm-lit $(LLVM_INSTALL_DIR)/bin/; \
		echo "Installing lit runtime into $(LIT_INSTALL_DIR)..."; \
		rm -rf "$(LIT_INSTALL_DIR)"; \
		mkdir -p "$(LLVM_INSTALL_DIR)/utils"; \
		cp -a "$(LIT_SOURCE_DIR)" "$(LIT_INSTALL_DIR)"; \
		find "$(LIT_INSTALL_DIR)" -type d -name "__pycache__" -prune -exec rm -rf {} + >/dev/null 2>&1 || true; \
	fi

lit-bootstrap:
	@if [ ! -d "$(LIT_SOURCE_DIR)/lit" ]; then \
		echo "Error: lit sources not found at $(LIT_SOURCE_DIR)."; \
		exit 1; \
	fi
	@if [ ! -f "$(LLVM_INSTALL_DIR)/bin/llvm-lit" ]; then \
		echo "Error: llvm-lit launcher not found at $(LLVM_INSTALL_DIR)/bin/llvm-lit."; \
		echo "Please run 'make llvm' first."; \
		exit 1; \
	fi
	@echo "Installing lit runtime into $(LIT_INSTALL_DIR)..."
	@rm -rf "$(LIT_INSTALL_DIR)"
	@mkdir -p "$(LLVM_INSTALL_DIR)/utils"
	@cp -a "$(LIT_SOURCE_DIR)" "$(LIT_INSTALL_DIR)"
	@find "$(LIT_INSTALL_DIR)" -type d -name "__pycache__" -prune -exec rm -rf {} + >/dev/null 2>&1 || true

llvm-lit: llvm
llvm-clean:
	rm -rf $(LLVM_BUILD_DIR)
	rm -f -r $(LLVM_INSTALL_DIR)

# ARTS
ARTS_BUILD_TYPE ?= Release
# Introspection
# Use ARTS_USE_COUNTERS=ON and ARTS_USE_METRICS=ON to enable introspection
ARTS_USE_COUNTERS ?= OFF
ARTS_USE_METRICS ?= OFF
# Logging level: 0=ERROR, 1=+WARN, 2=+INFO, 3=+DEBUG
ARTS_LOG_LEVEL ?= 1
# Counter configuration profile (defaults to timing-only for minimal overhead)
# Available profiles: profile-none.cfg, profile-timing.cfg, profile-workload.cfg, profile-overhead.cfg, profile-thread-edt.cfg
COUNTER_CONFIG_PATH ?= external/carts-benchmarks/configs/profiles/profile-timing.cfg
COUNTER_CONFIG_ABSPATH := $(abspath $(COUNTER_CONFIG_PATH))
# jemalloc allocator (opt-in). OFF because ARTS's jemalloc uses an empty
# symbol prefix, which crashes libarts on macOS via cross-allocator free.
ARTS_USE_JEMALLOC ?= OFF
# RDMA/RoCE transport via ARTS RSockets. OFF by default so local developer
# builds keep working on systems without librdmacm/RoCE devices.
ARTS_USE_RDMA ?= OFF

# Configuration hash file for ARTS build caching
ARTS_CONFIG_HASH_FILE := $(ARTS_BUILD_DIR)/.arts-build-config

# Hash the content of counter.cfg to detect changes
COUNTER_CONFIG_HASH := $(shell md5sum "$(COUNTER_CONFIG_ABSPATH)" 2>/dev/null | cut -d' ' -f1 || echo "no-config")

# Compute current configuration as a string for hashing
ARTS_CONFIG_STRING := $(ARTS_BUILD_TYPE)|$(ARTS_USE_COUNTERS)|$(ARTS_USE_METRICS)|$(ARTS_LOG_LEVEL)|$(COUNTER_CONFIG_ABSPATH)|$(COUNTER_CONFIG_HASH)|$(CARTS_LINKER_PATH)|$(ARTS_USE_JEMALLOC)|$(ARTS_USE_RDMA)

arts-download:
	@if [ ! -d "$(ARTS_DIR)/.git" ]; then \
		echo "Initializing ARTS submodule..."; \
		git submodule update --init --recursive external/arts; \
	else \
		echo "ARTS submodule already initialized."; \
	fi
arts:
	@if [ "$(ARTS_USE_JEMALLOC)" = "ON" ] && [ ! -f "$(ARTS_DIR)/third_party/jemalloc/autogen.sh" ]; then \
		echo "Initializing jemalloc submodule..."; \
		cd $(ARTS_DIR) && git submodule update --init --depth 1 third_party/jemalloc; \
	fi
	@$(call ensure_ninja_build_dir,$(ARTS_BUILD_DIR))
	@mkdir -p $(ARTS_BUILD_DIR); \
	$(call ensure_output_dirs); \
	mkdir -p $(ARTS_INSTALL_DIR); \
	CURRENT_HASH=$$(echo "$(ARTS_CONFIG_STRING)" | shasum -a 256 | cut -d' ' -f1); \
	STORED_HASH=""; \
	if [ -f "$(ARTS_CONFIG_HASH_FILE)" ]; then \
		STORED_HASH=$$(cat "$(ARTS_CONFIG_HASH_FILE)"); \
	fi; \
	if [ "$$CURRENT_HASH" = "$$STORED_HASH" ] && [ -f "$(ARTS_BUILD_DIR)/build.ninja" ]; then \
		echo "ARTS configuration unchanged, skipping cmake..."; \
	else \
		echo "Building ARTS (build_type=$(ARTS_BUILD_TYPE), counters=$(ARTS_USE_COUNTERS), metrics=$(ARTS_USE_METRICS), log_level=$(ARTS_LOG_LEVEL), counter_config=$(notdir $(COUNTER_CONFIG_PATH)), rdma=$(ARTS_USE_RDMA))..."; \
		$(CMAKE_CMD) -B $(ARTS_BUILD_DIR) -S $(ARTS_DIR) -G Ninja \
			-DCMAKE_C_COMPILER=$(LLVM_INSTALL_DIR)/bin/clang \
			-DCMAKE_CXX_COMPILER=$(LLVM_INSTALL_DIR)/bin/clang++ \
			-DCMAKE_BUILD_TYPE=$(ARTS_BUILD_TYPE) \
			-DARTS_LOG_LEVEL=$(ARTS_LOG_LEVEL) \
			-DARTS_USE_GPU=OFF \
			-DARTS_USE_JEMALLOC=$(ARTS_USE_JEMALLOC) \
			-DARTS_USE_RDMA=$(ARTS_USE_RDMA) \
			-DARTS_BUILD_BENCHMARKS=OFF \
			-DARTS_BUILD_TESTS=OFF \
			-DARTS_BUILD_EXAMPLES=OFF \
			-DCOUNTER_CONFIG_PATH="$(COUNTER_CONFIG_ABSPATH)" \
			-DCMAKE_INSTALL_PREFIX=$(ARTS_INSTALL_DIR) \
			$(LLVM_RUNTIME_CMAKE_FLAGS) \
			-DCMAKE_EXPORT_COMPILE_COMMANDS=ON; \
		echo "$$CURRENT_HASH" > "$(ARTS_CONFIG_HASH_FILE)"; \
	fi; \
	ninja $(NINJA_FLAGS) -C $(ARTS_BUILD_DIR) install;
arts-clean:
	rm -f -r $(ARTS_BUILD_DIR)
	rm -f -r $(ARTS_INSTALL_DIR)
	# rm -f -r $(ARTS_DIR)

# CARTS
build:
	@if [ ! -f "$(LLVM_INSTALL_DIR)/bin/clang" ] || [ ! -d "$(LLVM_BUILD_DIR)/lib/cmake/mlir" ]; then \
		echo "Error: LLVM is not built. Please run 'make llvm' or 'carts build --llvm' first."; \
		exit 1; \
	fi
	@if [ ! -d "$(POLYGEIST_BUILD_DIR)" ]; then \
		echo "Error: Polygeist is not built. Please run 'make polygeist' or 'carts build --polygeist' first."; \
		exit 1; \
	fi
	@$(call ensure_ninja_build_dir,$(CARTS_BUILD_DIR))
	$(call ensure_output_dirs)
	mkdir -p $(CARTS_BUILD_DIR)
	mkdir -p $(CARTS_INSTALL_DIR)
	$(CMAKE_CMD) -B $(CARTS_BUILD_DIR) \
		-S $(CARTS_DIR) -G Ninja \
		-DCMAKE_INSTALL_PREFIX=$(CARTS_INSTALL_DIR) \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_C_COMPILER=$(LLVM_INSTALL_DIR)/bin/clang \
		-DCMAKE_CXX_COMPILER=$(LLVM_INSTALL_DIR)/bin/clang++ \
		-DMLIR_DIR=$(LLVM_BUILD_DIR)/lib/cmake/mlir \
		-DClang_DIR=$(LLVM_BUILD_DIR)/lib/cmake/clang \
		-DPOLYGEIST_BUILD_DIR=$(POLYGEIST_BUILD_DIR) \
		-DPOLYGEIST_DIR=$(POLYGEIST_DIR) \
		$(LLVM_RUNTIME_CMAKE_FLAGS) \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON
	$(CMAKE_CMD) --build $(CARTS_BUILD_DIR) $(CMAKE_BUILD_FLAGS)
	$(CMAKE_CMD) --install $(CARTS_BUILD_DIR)

# Build only carts-compile
carts-compile-only:
	@if [ ! -d "$(CARTS_BUILD_DIR)" ]; then \
		echo "CARTS build directory not found. Run 'make build' first."; \
		exit 1; \
	fi
	# Force rebuild carts-compile even if dependencies haven't changed
	ninja $(NINJA_FLAGS) -C $(CARTS_BUILD_DIR) -t clean carts-compile
	ninja $(NINJA_FLAGS) -C $(CARTS_BUILD_DIR) carts-compile
	$(CMAKE_CMD) --install $(CARTS_BUILD_DIR) --component carts-compile

install: arts-download polygeist-download llvm-lit arts polygeist build

uninstall:
	cat $(CARTS_BUILD_DIR)/install_manifest.txt | xargs rm -f -r
	rm -rf $(CARTS_BUILD_DIR)

fulluninstall: uninstall arts-clean
	rm -f .arts

clean:
	@if [ -d "$(CARTS_BUILD_DIR)" ]; then \
		make -C $(CARTS_BUILD_DIR) clean -j || true; \
	fi
	rm -rf $(CARTS_BUILD_DIR)

check-doc-flags:
	python3 tools/scripts/check_doc_flags.py

.PHONY: all build install uninstall fulluninstall clean arts-download polygeist-download llvm llvm-lit lit-bootstrap check-doc-flags
