# Image names
CONTAINER_COMMAND := docker
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

.PHONY: all
all: build-builder

.PHONY: get-rust
get-rust:
	curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > scripts/install-rust.sh

.PHONY: build-base
build-base:
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.base -t $(BASE_IMAGE) . \
  		--build-arg UBUNTU_TAG=$(grep UBUNTU_TAG $(CONTAINER_DIR)/digests.env | cut -d= -f2)
  		--build-arg UBUNTU_DIGEST=$(grep UBUNTU_DIGEST $(CONTAINER_DIR)/digests.env | cut -d= -f2)

.PHONY: build-base-builder
build-base-builder:
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.base-builder -t $(BASE_BUILDER_IMAGE) .

.PHONY: build-builder
build-builder: build-base build-base-builder
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.builder -t $(BUILDER_IMAGE) .

.PHONY: build-base-runner
build-base-runner:
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.base-runner -t $(BASE_RUNNER_IMAGE) .

.PHONY: build-runner
build-runner:
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.runner -t $(RUNNER_IMAGE) . \
  		--build-arg BASE_IMAGE_DIGEST=$(grep BASE_IMAGE_DIGEST digests.env | cut -d= -f2)

# Build everything (may be slow)
.PHONY: build-images
build-images: build-base build-base-builder build-builder build-base-runner build-runner

.PHONY: run-builder
run-builder:
	docker run --rm \
		-v $(PWD)/build:/build/target/release \
		-v $(PWD)/app:/build \
		$(BUILDER_IMAGE)

.PHONY: run-app
run-app:
	docker run -d $(RUNNER_IMAGE)

.PHONY: clean-base
clean-base:
	$(CONTAINER_COMMAND) rmi $(BASE_IMAGE) || true

.PHONY: clean-base-builder
clean-base-builder:
	$(CONTAINER_COMMAND) rmi $(BASE_BUILDER_IMAGE) || true

.PHONY: clean-base-runner
clean-base-runner:
	$(CONTAINER_COMMAND) rmi $(BASE_RUNNER_IMAGE) || true

.PHONY: clean-runner
clean-runner:
	$(CONTAINER_COMMAND) rmi $(RUNNER_IMAGE) || true

.PHONY: clean-all
clean-all: clean-base clean-base-builder clean-builder clean-base-runner clean-runner
