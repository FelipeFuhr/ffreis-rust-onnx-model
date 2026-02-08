FROM ffreis/base-builder

WORKDIR /build

COPY app/ .


RUN cargo test --verbosr

ENTRYPOINT ["cargo", "build"]
CMD ["--release"]