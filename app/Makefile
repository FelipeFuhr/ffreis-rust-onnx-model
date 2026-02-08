.DEFAULT_GOAL := help

SHELL := /usr/bin/env bash
CARGO ?= cargo
COVERAGE_MIN ?= 90

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
	$(CARGO) llvm-cov --all-features --cobertura --output-path ../coverage/cobertura.xml

.PHONY: coverage-check
coverage-check: ## Fail if coverage is below COVERAGE_MIN
	@command -v cargo-llvm-cov >/dev/null 2>&1 || { \
		echo "cargo-llvm-cov not installed. Install with: cargo install cargo-llvm-cov --locked"; \
		exit 1; \
	}
	$(MAKE) coverage
	python3 - <<'PY'
import os
import sys
import xml.etree.ElementTree as ET

min_cov = float(os.environ.get("COVERAGE_MIN", "90"))
path = os.path.join("..", "coverage", "cobertura.xml")
tree = ET.parse(path)
root = tree.getroot()
rate = float(root.attrib.get("line-rate", "0")) * 100.0
if rate + 1e-9 < min_cov:
    print(f"Coverage {rate:.2f}% is below minimum {min_cov:.2f}%")
    sys.exit(1)
print(f"Coverage {rate:.2f}% meets minimum {min_cov:.2f}%")
PY

.PHONY: clean
clean: ## Clean build artifacts
	$(CARGO) clean
