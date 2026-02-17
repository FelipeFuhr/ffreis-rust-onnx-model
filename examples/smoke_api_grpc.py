from __future__ import annotations

import json
import os
import time
import urllib.request

import grpc


def _wait_http_ok(url: str, timeout_seconds: float = 30.0) -> bytes:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3.0) as response:  # noqa: S310
                if response.status == 200:
                    return response.read()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"timed out waiting for HTTP 200 at {url}: {last_error}")


def _assert_http(api_base: str) -> None:
    for path in ("/live", "/healthz", "/ready", "/readyz", "/ping"):
        _ = _wait_http_ok(f"{api_base}{path}")

    req = urllib.request.Request(
        f"{api_base}/invocations",
        data=json.dumps({"instances": [[1.0, 2.0], [3.0, 4.0]]}).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5.0) as response:  # noqa: S310
        assert response.status == 200
        payload = json.loads(response.read().decode("utf-8"))
    assert payload.get("predictions") == [0, 0], payload


def _assert_grpc(target: str) -> None:
    with grpc.insecure_channel(target) as channel:
        live_rpc = channel.unary_unary(
            "/onnxserving.grpc.InferenceService/Live",
            request_serializer=lambda _: b"",
            response_deserializer=lambda data: data,
        )
        ready_rpc = channel.unary_unary(
            "/onnxserving.grpc.InferenceService/Ready",
            request_serializer=lambda _: b"",
            response_deserializer=lambda data: data,
        )
        _ = live_rpc(b"", timeout=5.0)
        _ = ready_rpc(b"", timeout=5.0)


def main() -> None:
    api_base = os.getenv("API_BASE", "http://serving-api:8080")
    grpc_target = os.getenv("GRPC_TARGET", "serving-grpc:50052")
    _assert_http(api_base)
    _assert_grpc(grpc_target)
    print("rust serving API+gRPC smoke passed")


if __name__ == "__main__":
    main()
