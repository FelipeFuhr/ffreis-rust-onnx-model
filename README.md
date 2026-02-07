# Incremental Docker Build with Rust

A minimal guide to building Rust applications with Docker using multi-stage builds for lightweight container images.

## What is this?

This project demonstrates a **multi-stage Docker build** that:
1. **Stage 1 (Builder)**: Installs Rust and compiles your application
2. **Stage 2 (Runtime)**: Copies only the compiled binary to a clean image

## Quick Start

### Build the image
```bash
make build
```
Or manually:
```bash
docker build -f container/Dockerfile -t ffreis-model .
```

### Run the container
```bash
docker run ffreis-model
```

## How it works

The **Dockerfile** has two stages:

**Builder Stage**: 
- Starts with `ubuntu:26.04`
- Installs Rust and build tools (gcc, curl, ca-certificates)
- Compiles your Rust app in release mode

**Runtime Stage**:
- Starts fresh with `ubuntu:26.04`
- Copies **only** the compiled binary from the builder
- Runs the binary

## Available Commands

```bash
make build              # Build the Docker image
make clean              # Remove the Docker image
make get-rust           # Download rustup installer script
```

## Why multi-stage builds?

- **Smaller images**: Only runtime dependencies in final image
- **Faster deployments**: Less data to push/pull
- **Cleaner separation**: Build environment != runtime environment
- **Security**: Reduces attack surface by removing compilers and dev tools
