use std::time::Duration;
use std::{fs, path::PathBuf};

use app::grpc::inference_service_client::InferenceServiceClient;
use app::grpc::{LiveRequest, PredictRequest, ReadyRequest};
use app::{serve_grpc, serve_http, AppConfig};
use tempfile::TempDir;
use tokio::net::TcpListener;

async fn start_http_server(
    cfg: AppConfig,
) -> (String, tokio::task::JoinHandle<Result<(), std::io::Error>>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral http port");
    let addr = listener.local_addr().expect("http local addr");
    let handle = tokio::spawn(serve_http(listener, cfg));
    (format!("http://{}", addr), handle)
}

async fn start_grpc_server(
    cfg: AppConfig,
) -> (
    String,
    tokio::task::JoinHandle<Result<(), Box<dyn std::error::Error + Send + Sync>>>,
) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral grpc port");
    let addr = listener.local_addr().expect("grpc local addr");
    let handle = tokio::spawn(serve_grpc(listener, cfg));
    (format!("http://{}", addr), handle)
}

fn build_cfg_with_temp_model() -> (TempDir, AppConfig) {
    let tmp = tempfile::tempdir().expect("temp dir");
    let model_path: PathBuf = tmp.path().join("model.onnx");
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("model.onnx");
    let fixture = fs::read(&fixture_path).expect("read fixture model");
    fs::write(&model_path, fixture).expect("write fixture model");
    let cfg = AppConfig {
        model_type: "onnx".to_string(),
        model_dir: tmp.path().to_string_lossy().to_string(),
        model_filename: "model.onnx".to_string(),
        max_records: 1000,
        ..AppConfig::default()
    };
    (tmp, cfg)
}

#[tokio::test]
async fn http_health_and_ready_endpoints_are_available() {
    let (_tmp, cfg) = build_cfg_with_temp_model();
    let (base_url, handle) = start_http_server(cfg).await;
    let client = reqwest::Client::new();

    // Wait for the server to become ready by polling /readyz with a timeout.
    let mut ready = false;
    for _ in 0u32..100 {
        match client
            .get(format!("{base_url}/readyz"))
            .send()
            .await
        {
            Ok(response) if response.status() == reqwest::StatusCode::OK => {
                ready = true;
                break;
            }
            _ => {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }
    }
    assert!(ready, "HTTP server did not become ready in time");

    for path in ["/live", "/healthz", "/ready", "/readyz", "/ping"] {
        let response = client
            .get(format!("{base_url}{path}"))
            .send()
            .await
            .expect("http endpoint should answer");
        assert_eq!(response.status(), reqwest::StatusCode::OK);
    }
    let metrics = client
        .get(format!("{base_url}/metrics"))
        .send()
        .await
        .expect("metrics endpoint should answer");
    assert_eq!(metrics.status(), reqwest::StatusCode::OK);

    handle.abort();
}

#[tokio::test]
async fn http_and_grpc_predict_parity_for_json_and_csv() {
    let (_tmp, cfg) = build_cfg_with_temp_model();
    let (http_base, http_handle) = start_http_server(cfg.clone()).await;
    let (grpc_url, grpc_handle) = start_grpc_server(cfg).await;

    let client = reqwest::Client::new();
    
    // Retry gRPC connection until it succeeds or timeout
    let mut grpc_client = None;
    for _ in 0u32..100 {
        match InferenceServiceClient::connect(grpc_url.clone()).await {
            Ok(client) => {
                grpc_client = Some(client);
                break;
            }
            Err(_) => {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }
    }
    let mut grpc_client = grpc_client.expect("gRPC client should connect after retries");

    let payloads = vec![
        (
            br#"{"instances":[[1.0,2.0],[3.0,4.0]]}"#.to_vec(),
            "application/json",
        ),
        (b"1,2\n3,4\n".to_vec(), "text/csv"),
    ];

    for (payload, content_type) in payloads {
        let http_response = client
            .post(format!("{http_base}/invocations"))
            .header("content-type", content_type)
            .header("accept", "application/json")
            .body(payload.clone())
            .send()
            .await
            .expect("http invocation response");
        assert_eq!(http_response.status(), reqwest::StatusCode::OK);
        assert!(http_response.headers().contains_key("x-trace-id"));
        assert!(http_response.headers().contains_key("x-span-id"));
        let http_body = http_response.bytes().await.expect("http bytes");

        let grpc_reply = grpc_client
            .predict(PredictRequest {
                payload,
                content_type: content_type.to_string(),
                accept: "application/json".to_string(),
            })
            .await
            .expect("grpc predict should pass")
            .into_inner();

        assert_eq!(grpc_reply.content_type, "application/json");
        assert_eq!(http_body.as_ref(), grpc_reply.body.as_slice());
    }

    let live = grpc_client
        .live(LiveRequest {})
        .await
        .expect("grpc live")
        .into_inner();
    assert!(live.ok);
    assert_eq!(live.status, "live");

    let ready = grpc_client
        .ready(ReadyRequest {})
        .await
        .expect("grpc ready")
        .into_inner();
    assert!(ready.ok);
    assert_eq!(ready.status, "ready");

    http_handle.abort();
    grpc_handle.abort();
}

#[tokio::test]
async fn invalid_json_maps_to_http_400_and_grpc_invalid_argument() {
    let (_tmp, cfg) = build_cfg_with_temp_model();
    let (http_base, http_handle) = start_http_server(cfg.clone()).await;
    let (grpc_url, grpc_handle) = start_grpc_server(cfg).await;

    let bad_payload = b"{not-json".to_vec();
    let client = reqwest::Client::new();
    let http_response = client
        .post(format!("{http_base}/invocations"))
        .header("content-type", "application/json")
        .header("accept", "application/json")
        .body(bad_payload.clone())
        .send()
        .await
        .expect("http bad response");
    assert_eq!(http_response.status(), reqwest::StatusCode::BAD_REQUEST);

    // Retry gRPC connection until it succeeds or timeout
    let mut grpc_client = None;
    for _ in 0u32..100 {
        match InferenceServiceClient::connect(grpc_url.clone()).await {
            Ok(client) => {
                grpc_client = Some(client);
                break;
            }
            Err(_) => {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }
    }
    let mut grpc_client = grpc_client.expect("gRPC client should connect after retries");
    let grpc_error = grpc_client
        .predict(PredictRequest {
            payload: bad_payload,
            content_type: "application/json".to_string(),
            accept: "application/json".to_string(),
        })
        .await
        .expect_err("grpc should reject invalid json");
    assert_eq!(grpc_error.code(), tonic::Code::InvalidArgument);

    http_handle.abort();
    grpc_handle.abort();
}

#[tokio::test]
async fn record_limit_violation_maps_to_http_400_and_grpc_invalid_argument() {
    let (_tmp, mut cfg) = build_cfg_with_temp_model();
    cfg.max_records = 1;
    let (http_base, http_handle) = start_http_server(cfg.clone()).await;
    let (grpc_url, grpc_handle) = start_grpc_server(cfg).await;
    tokio::time::sleep(Duration::from_millis(75)).await;

    let payload = br#"{"instances":[[1.0,2.0],[3.0,4.0]]}"#.to_vec();
    let client = reqwest::Client::new();
    let http_response = client
        .post(format!("{http_base}/invocations"))
        .header("content-type", "application/json")
        .header("accept", "application/json")
        .body(payload.clone())
        .send()
        .await
        .expect("http response");
    assert_eq!(http_response.status(), reqwest::StatusCode::BAD_REQUEST);
    let body: serde_json::Value = http_response.json().await.expect("json body");
    assert!(body["error"]
        .as_str()
        .unwrap_or_default()
        .contains("too_many_records"));

    // Retry gRPC connection until it succeeds or timeout
    let mut grpc_client = None;
    for _ in 0u32..100 {
        match InferenceServiceClient::connect(grpc_url.clone()).await {
            Ok(client) => {
                grpc_client = Some(client);
                break;
            }
            Err(_) => {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }
    }
    let mut grpc_client = grpc_client.expect("gRPC client should connect after retries");
    let grpc_error = grpc_client
        .predict(PredictRequest {
            payload,
            content_type: "application/json".to_string(),
            accept: "application/json".to_string(),
        })
        .await
        .expect_err("grpc should reject record overflow");
    assert_eq!(grpc_error.code(), tonic::Code::InvalidArgument);
    assert!(grpc_error.message().contains("too_many_records"));

    http_handle.abort();
    grpc_handle.abort();
}

#[tokio::test]
async fn payload_too_large_returns_413() {
    let (_tmp, mut cfg) = build_cfg_with_temp_model();
    cfg.max_body_bytes = 8;
    let (http_base, http_handle) = start_http_server(cfg).await;
    tokio::time::sleep(Duration::from_millis(75)).await;

    let client = reqwest::Client::new();
    let response = client
        .post(format!("{http_base}/invocations"))
        .header("content-type", "application/json")
        .header("accept", "application/json")
        .body(br#"{"instances":[[1.0,2.0],[3.0,4.0]]}"#.to_vec())
        .send()
        .await
        .expect("http response");
    assert_eq!(response.status(), reqwest::StatusCode::PAYLOAD_TOO_LARGE);

    http_handle.abort();
}
