use std::env;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct AppConfig {
    pub service_name: String,
    pub service_version: String,
    pub deployment_env: String,
    pub model_dir: String,
    pub model_type: String,
    pub model_filename: String,
    pub input_mode: String,
    pub default_content_type: String,
    pub default_accept: String,
    pub tabular_dtype: String,
    pub csv_delimiter: String,
    pub csv_has_header: String,
    pub csv_skip_blank_lines: bool,
    pub json_key_instances: String,
    pub jsonl_features_key: String,
    pub tabular_id_columns: String,
    pub tabular_feature_columns: String,
    pub predictions_only: bool,
    pub json_output_key: String,
    pub max_body_bytes: usize,
    pub max_records: usize,
    pub max_inflight: usize,
    pub acquire_timeout_s: f64,
    pub prometheus_enabled: bool,
    pub prometheus_path: String,
    pub otel_enabled: bool,
    pub otel_endpoint: String,
    pub otel_headers: String,
    pub otel_timeout_s: f64,
    pub onnx_input_map_json: String,
    pub onnx_output_map_json: String,
    pub onnx_input_dtype_map_json: String,
    pub onnx_dynamic_batch: bool,
    pub tabular_num_features: usize,
    pub onnx_input_name: String,
    pub onnx_output_name: String,
    pub onnx_output_index: usize,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            service_name: env_str("SERVICE_NAME", "model-serving-universal"),
            service_version: env_str("SERVICE_VERSION", "dev"),
            deployment_env: env_str("DEPLOYMENT_ENV", "local"),
            model_dir: env_str("SM_MODEL_DIR", "/opt/ml/model"),
            model_type: env_str("MODEL_TYPE", "").trim().to_ascii_lowercase(),
            model_filename: env_str("MODEL_FILENAME", "").trim().to_string(),
            input_mode: env_str("INPUT_MODE", "tabular").trim().to_ascii_lowercase(),
            default_content_type: env_str("DEFAULT_CONTENT_TYPE", "application/json"),
            default_accept: env_str("DEFAULT_ACCEPT", "application/json"),
            tabular_dtype: env_str("TABULAR_DTYPE", "float32")
                .trim()
                .to_ascii_lowercase(),
            csv_delimiter: env_str("CSV_DELIMITER", ","),
            csv_has_header: env_str("CSV_HAS_HEADER", "auto")
                .trim()
                .to_ascii_lowercase(),
            csv_skip_blank_lines: env_bool("CSV_SKIP_BLANK_LINES", true),
            json_key_instances: env_str("JSON_KEY_INSTANCES", "instances"),
            jsonl_features_key: env_str("JSONL_FEATURES_KEY", "features"),
            tabular_id_columns: env_str("TABULAR_ID_COLUMNS", "").trim().to_string(),
            tabular_feature_columns: env_str("TABULAR_FEATURE_COLUMNS", "").trim().to_string(),
            predictions_only: env_bool("RETURN_PREDICTIONS_ONLY", true),
            json_output_key: env_str("JSON_OUTPUT_KEY", "predictions"),
            max_body_bytes: env_usize("MAX_BODY_BYTES", 6 * 1024 * 1024),
            max_records: env_usize("MAX_RECORDS", 5000),
            max_inflight: env_usize("MAX_INFLIGHT", 16),
            acquire_timeout_s: env_f64("ACQUIRE_TIMEOUT_S", 0.25),
            prometheus_enabled: env_bool("PROMETHEUS_ENABLED", true),
            prometheus_path: env_str("PROMETHEUS_PATH", "/metrics"),
            otel_enabled: env_bool("OTEL_ENABLED", true),
            otel_endpoint: env_str("OTEL_EXPORTER_OTLP_ENDPOINT", "")
                .trim()
                .to_string(),
            otel_headers: env_str("OTEL_EXPORTER_OTLP_HEADERS", ""),
            otel_timeout_s: env_f64("OTEL_EXPORTER_OTLP_TIMEOUT", 10.0),
            onnx_input_map_json: env_str("ONNX_INPUT_MAP_JSON", "").trim().to_string(),
            onnx_output_map_json: env_str("ONNX_OUTPUT_MAP_JSON", "").trim().to_string(),
            onnx_input_dtype_map_json: env_str("ONNX_INPUT_DTYPE_MAP_JSON", "").trim().to_string(),
            onnx_dynamic_batch: env_bool("ONNX_DYNAMIC_BATCH", true),
            tabular_num_features: env_usize("TABULAR_NUM_FEATURES", 0),
            onnx_input_name: env_str("ONNX_INPUT_NAME", "").trim().to_string(),
            onnx_output_name: env_str("ONNX_OUTPUT_NAME", "").trim().to_string(),
            onnx_output_index: env_usize("ONNX_OUTPUT_INDEX", 0),
        }
    }
}

fn env_str(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}

fn env_bool(name: &str, default: bool) -> bool {
    match env::var(name) {
        Ok(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "y" | "on"
        ),
        Err(_) => default,
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(default)
}

impl AppConfig {
    pub(crate) fn model_path(&self) -> PathBuf {
        let dir = Path::new(&self.model_dir);
        if !dir.exists() {
            panic!(
                "Configured model directory '{}' does not exist",
                self.model_dir
            );
        }
        if !dir.is_dir() {
            panic!(
                "Configured model path '{}' is not a directory",
                self.model_dir
            );
        }
        let filename = if self.model_filename.trim().is_empty() {
            "model.onnx".to_string()
        } else {
            self.model_filename.clone()
        };
        dir.join(filename)
    }
}
