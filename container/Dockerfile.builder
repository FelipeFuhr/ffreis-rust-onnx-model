FROM ffreis/base-builder

WORKDIR /build

COPY app/ .

RUN cargo test --verbose

ENTRYPOINT ["cargo", "build"]
CMD ["--release"]
