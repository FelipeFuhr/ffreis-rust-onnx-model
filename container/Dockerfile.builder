FROM ffreis/base-builder

WORKDIR /build

COPY app/ .

ENTRYPOINT ["cargo", "build"]
CMD ["--release"]