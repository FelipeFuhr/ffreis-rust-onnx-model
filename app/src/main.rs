use std::env;

use app::{run_grpc_server, run_http_server, AppConfig};

#[tokio::main]
async fn main() {
    let mode = env::var("SERVE_MODE").unwrap_or_else(|_| "http".to_string());
    let host = env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port = env::var("PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(if mode == "grpc" { 50052 } else { 8080 });
    let cfg = AppConfig::default();

    match mode.as_str() {
        "grpc" => {
            if let Err(err) = run_grpc_server(&host, port, cfg).await {
                eprintln!("grpc server failed: {err}");
                std::process::exit(1);
            }
        }
        _ => {
            if let Err(err) = run_http_server(&host, port, cfg).await {
                eprintln!("http server failed: {err}");
                std::process::exit(1);
            }
        }
    }
}
