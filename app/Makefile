.DEFAULT_GOAL := help

SHELL := /usr/bin/env bash
CARGO ?= cargo
COVERAGE_MIN ?= 90
ROOT_DIR ?= ..
COVERAGE_XML ?= $(ROOT_DIR)/coverage/cobertura.xml
CHECK_COVERAGE_SCRIPT ?= $(ROOT_DIR)/scripts/check_coverage.py
PYTHON ?= python3

.PHONY: help
help: ## Show help
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: fmt
fmt: ## Format Rust code
	$(CARGO) fmt --all

.PHONY: fmt-check
fmt-check: ## Check Rust formatting
	$(CARGO) fmt --all -- --check

.PHONY: clippy
clippy: ## Run clippy lints
	$(CARGO) clippy --all-targets --all-features

.PHONY: lint
lint: fmt-check clippy ## Run format check and clippy

.PHONY: test
test: ## Run tests
	$(CARGO) test --verbose

.PHONY: build
build: ## Build the app
	$(CARGO) build --verbose

.PHONY: coverage
coverage: ## Generate coverage report (Cobertura XML) into ../coverage/
	@command -v cargo-llvm-cov >/dev/null 2>&1 || { \
		echo "cargo-llvm-cov not installed. Install with: cargo install cargo-llvm-cov --locked"; \
		exit 1; \
	}
	mkdir -p ../coverage
	$(CARGO) llvm-cov --all-features --cobertura --output-path $(COVERAGE_XML)

.PHONY: coverage-check
coverage-check: ## Fail if coverage is below COVERAGE_MIN
	@command -v cargo-llvm-cov >/dev/null 2>&1 || { \
		echo "cargo-llvm-cov not installed. Install with: cargo install cargo-llvm-cov --locked"; \
		exit 1; \
	}
	$(MAKE) coverage
	@COVERAGE_MIN="$(COVERAGE_MIN)" COVERAGE_XML="$(COVERAGE_XML)" \
		$(PYTHON) $(CHECK_COVERAGE_SCRIPT)