FROM ffreis/base-builder

RUN getent group appgroup >/dev/null || groupadd -g 10001 appgroup \
    && id -u appuser >/dev/null 2>&1 || useradd -u 10001 -g appgroup -m -s /usr/sbin/nologin appuser

WORKDIR /build

COPY --chown=appuser:appgroup app/ .

USER appuser:appgroup

RUN cargo test --verbose

ENTRYPOINT ["cargo", "build"]
CMD ["--release"]
