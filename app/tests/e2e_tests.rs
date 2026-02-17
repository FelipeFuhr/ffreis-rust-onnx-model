use std::net::TcpListener;
use std::process::Command;
use std::thread;
use std::time::Duration;

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    listener.local_addr().expect("local addr").port()
}

#[test]
fn binary_starts_http_service_and_answers_healthz() {
    let Some(exe) = option_env!("CARGO_BIN_EXE_app") else {
        return; // Skip when binary is not available in this context.
    };
    let port = free_port();
    let mut child = Command::new(exe)
        .env("SERVE_MODE", "http")
        .env("HOST", "127.0.0.1")
        .env("PORT", port.to_string())
        .spawn()
        .expect("spawn app binary");

    thread::sleep(Duration::from_millis(250));
    let url = format!("http://127.0.0.1:{port}/healthz");
    let response = reqwest::blocking::get(url).expect("healthz request");
    assert_eq!(response.status(), reqwest::StatusCode::OK);

    let _ = child.kill();
    let _ = child.wait();
}
