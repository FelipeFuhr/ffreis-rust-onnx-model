FROM ffreis/base-builder

RUN groupadd -g 10001 appgroup \
    && useradd -u 10001 -g appgroup -m -s /usr/sbin/nologin appuser

WORKDIR /build

COPY app/ .

RUN chown -R appuser:appgroup /build

USER appuser:appgroup

RUN cargo test --verbose

ENTRYPOINT ["cargo", "build"]
CMD ["--release"]
