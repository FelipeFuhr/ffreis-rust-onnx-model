.PHONY: all
all: build

.PHONY: get-rust
get-rust:
	curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > scripts/install-rust.sh

.PHONY: build
build:
	docker build -f container/Dockerfile -t ffreis-model .;

.PHONY: clean
clean:
	docker rmi ffreis-model || true
