# Builder
FROM ubuntu:26.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

COPY scripts/install-rust.sh /tmp/install-rust.sh

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    gcc \
    libc6-dev && \
    sh /tmp/install-rust.sh -y && \
    rm /tmp/install-rust.sh && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=/root/.cargo/bin:$PATH

COPY app /build
WORKDIR /build

RUN cargo build --release

# Runner
FROM ubuntu:26.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /run

COPY --from=builder /build/target/release /run

CMD ["./hello_rust"]

# COPY debs/deb-manifest.json /debs/deb-manifest.json
# COPY debs/debs-sha256.txt /debs/debs-sha256.txt
# COPY debs/*.deb /debs/ 2>/dev/null || true
# RUN if [ -f /debs/debs-sha256.txt ]; then \
#             cd /debs && sha256sum -c debs-sha256.txt || true; \
#             dpkg -i ./*.deb || apt-get -f install -y; \
#             rm -rf /debs; \
#         else \
#             apt-get update && apt-get install -y --no-install-recommends ca-certificates curl libssl-dev && rm -rf /var/lib/apt/lists/*; \
#         fi

# # copy the built binary
# COPY --from=builder /work/rust-app/target/release/onnx_api /usr/local/bin/onnx_api

# EXPOSE 8080
