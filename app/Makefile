.DEFAULT_GOAL := help

SHELL := /usr/bin/env bash
CARGO ?= cargo

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

.PHONY: coverage
coverage: ## Generate coverage report (Cobertura XML) into ../coverage/
	@command -v cargo-llvm-cov >/dev/null 2>&1 || { \
		echo "cargo-llvm-cov not installed. Install with: cargo install cargo-llvm-cov --locked"; \
		exit 1; \
	}
	mkdir -p ../coverage
	$(CARGO) llvm-cov --all-features --cobertura --output-path ../coverage/cobertura.xml

.PHONY: clean
clean: ## Clean build artifacts
	$(CARGO) clean
