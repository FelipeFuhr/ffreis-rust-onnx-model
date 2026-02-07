FROM ffreis/base-builder

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /build

COPY app/ .

ENTRYPOINT ["cargo", "build"]
CMD ["--release"]