use axum::http::HeaderName;
use axum::response::Response as AxumResponse;
use uuid::Uuid;

use crate::config::AppConfig;

pub fn init_telemetry(cfg: &AppConfig) -> Result<(), String> {
    if !cfg.otel_enabled {
        return Ok(());
    }
    if cfg.otel_endpoint.trim().is_empty() {
        eprintln!(
            "OTEL_ENABLED=true but OTEL_EXPORTER_OTLP_ENDPOINT is empty; tracing exporter is disabled"
        );
    }
    Ok(())
}

pub(crate) fn attach_trace_correlation_headers(response: &mut AxumResponse) {
    let trace_id = Uuid::new_v4().simple().to_string();
    let span_id = &trace_id[..16];
    if let Ok(value) = trace_id.parse() {
        response
            .headers_mut()
            .insert(HeaderName::from_static("x-trace-id"), value);
    }
    if let Ok(value) = span_id.parse() {
        response
            .headers_mut()
            .insert(HeaderName::from_static("x-span-id"), value);
    }
}
