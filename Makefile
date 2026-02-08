.DEFAULT_GOAL := help

# Setup
SHELL := /usr/bin/env bash
CARGO ?= cargo
CONTAINER_COMMAND := podman
LEFTHOOK ?= $(or $(wildcard $(LEFTHOOK_BIN)),lefthook)

# Image names
PREFIX:=ffreis
BASE_DIR := .
CONTAINER_DIR := container

BASE_IMAGE:=$(PREFIX)/base
BASE_BUILDER_IMAGE:=$(PREFIX)/base-builder
BUILDER_IMAGE:=$(PREFIX)/builder
BASE_RUNNER_IMAGE:=$(PREFIX)/base-runner
RUNNER_IMAGE:=$(PREFIX)/runner

# Extract app name from Cargo.toml
APP_NAME=$(shell grep '^name' app/Cargo.toml | sed 's/name[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/')

# Extract digests from digests.env
BASE_IMAGE_VALUE=$(shell grep '^BASE_IMAGE=' $(CONTAINER_DIR)/digests.env | cut -d= -f2)
BASE_DIGEST_VALUE=$(shell grep '^BASE_DIGEST=' $(CONTAINER_DIR)/digests.env | cut -d= -f2)

.PHONY: help
help: ## Show help
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: all
all: lint build run coverage

.PHONY: get-rust
get-rust:
	curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > scripts/install-rust.sh

.PHONY: build-base
build-base:
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.base -t $(BASE_IMAGE) . \
		--build-arg BASE_IMAGE=$(BASE_IMAGE_VALUE) \
		--build-arg BASE_DIGEST=$(BASE_DIGEST_VALUE)

.PHONY: build-base-builder
build-base-builder: get-rust
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.base-builder -t $(BASE_BUILDER_IMAGE) .

.PHONY: build-builder
build-builder: build-base build-base-builder
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.builder -t $(BUILDER_IMAGE) .

.PHONY: build-base-runner
build-base-runner:
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.base-runner -t $(BASE_RUNNER_IMAGE) .

.PHONY: build-runner
build-runner: build-base-runner run-builder
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.runner -t $(RUNNER_IMAGE) . \
		--build-arg APP_NAME=$(APP_NAME)

# Build everything (may be slow)
.PHONY: build-images
build-images: get-rust build-base build-base-builder build-builder build-base-runner

.PHONY: run-builder
run-builder: build-builder
	$(CONTAINER_COMMAND) run --rm \
		-v $(PWD)/build:/build/target/release \
		-v $(PWD)/app:/build \
		$(BUILDER_IMAGE)

.PHONY: build
build: build-images run-builder build-runner

.PHONY: build-app-local
build-app-local:
	$(MAKE) -C app build

.PHONY: run-app
run-app:
	$(CONTAINER_COMMAND) run $(RUNNER_IMAGE)

.PHONY: run
run: run-app

.PHONY: fmt
fmt:
	$(MAKE) -C app fmt

.PHONY: fmt-check
fmt-check:
	$(MAKE) -C app fmt-check

.PHONY: clippy
clippy:
	$(MAKE) -C app clippy

.PHONY: test
test:
	$(MAKE) -C app test

.PHONY: lint
lint: fmt-check clippy

.PHONY: coverage
coverage: ## Generate coverage report (Cobertura XML) into ./coverage/
	$(MAKE) -C app coverage

.PHONY: coverage-check
coverage-check: ## Fail if coverage is below COVERAGE_MIN
	$(MAKE) -C app coverage-check

.PHONY: lefthook-bootstrap
lefthook-bootstrap: ## Download lefthook locally into ./.bin
	@mkdir -p .bin
	@if [ ! -x "$(LEFTHOOK_BIN)" ]; then \
		echo "Downloading lefthook $(LEFTHOOK_VERSION)..." ; \
		curl -fsSL -o "$(LEFTHOOK_BIN)" \
		  "https://github.com/evilmartians/lefthook/releases/download/v$(LEFTHOOK_VERSION)/lefthook_$(LEFTHOOK_VERSION)_Linux_x86_64"; \
		chmod +x "$(LEFTHOOK_BIN)"; \
	fi

.PHONY: lefthook-install
lefthook-install: lefthook-bootstrap
	@if command -v lefthook >/dev/null 2>&1; then \
		lefthook install; \
	else \
		echo "lefthook not found. Install it or set LEFTHOOK_BIN."; \
		exit 1; \
	fi

.PHONY: lefthook-run
lefthook-run:
	@$(LEFTHOOK) run pre-commit
	@$(LEFTHOOK) run pre-push

.PHONY: lefthook
lefthook: lefthook-install lefthook-run

.PHONY: clean-app
clean-app:
	$(MAKE) -C app clean

.PHONY: clean-repo
clean-repo: clean-app
	rm -rf build coverage
	rm -f scripts/install-rust.sh

.PHONY: clean-base
clean-base:
	$(CONTAINER_COMMAND) rmi $(BASE_IMAGE) || true

.PHONY: clean-base-builder
clean-base-builder:
	$(CONTAINER_COMMAND) rmi $(BASE_BUILDER_IMAGE) || true

.PHONY: clean-builder
clean-builder:
	$(CONTAINER_COMMAND) rmi $(BUILDER_IMAGE) || true

.PHONY: clean-base-runner
clean-base-runner:
	$(CONTAINER_COMMAND) rmi $(BASE_RUNNER_IMAGE) || true

.PHONY: clean-runner
clean-runner:
	$(CONTAINER_COMMAND) rmi $(RUNNER_IMAGE) || true

.PHONY: clean-all
clean-all: clean-app clean-repo clean-base clean-base-builder clean-builder clean-base-runner clean-runner

