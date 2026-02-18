use std::collections::HashMap;
use std::env;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::header::{ACCEPT, CONTENT_TYPE};
use axum::http::{HeaderMap, HeaderName, StatusCode};
use axum::response::{IntoResponse, Response as AxumResponse};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio::sync::{RwLock, Semaphore};
use tokio::time::timeout;
use tonic::transport::Server;
use tonic::{Code, Request, Response, Status};
use tract_onnx::prelude::{tvec, Framework, InferenceModelExt, RunnableModel, TypedFact, TypedOp};
use uuid::Uuid;

pub mod grpc {
    tonic::include_proto!("onnxserving.grpc");
}

const JSON_CONTENT_TYPES: &[&str] = &["application/json", "application/*+json"];
const JSON_LINES_CONTENT_TYPES: &[&str] = &[
    "application/jsonlines",
    "application/x-jsonlines",
    "application/jsonl",
    "application/x-ndjson",
];
const CSV_CONTENT_TYPES: &[&str] = &["text/csv", "application/csv"];
const SAGEMAKER_CONTENT_TYPE_HEADER: &str = "x-amzn-sagemaker-content-type";
const SAGEMAKER_ACCEPT_HEADER: &str = "x-amzn-sagemaker-accept";

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
    fn model_path(&self) -> PathBuf {
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

#[derive(Clone, Debug)]
pub struct ParsedInput {
    pub x: Option<Vec<Vec<f64>>>,
    pub tensors: Option<HashMap<String, Value>>,
    pub meta: Option<Value>,
}

impl ParsedInput {
    fn batch_size(&self) -> Result<usize, String> {
        if let Some(x) = &self.x {
            return Ok(x.len());
        }
        if let Some(tensors) = &self.tensors {
            let mut inferred: Vec<usize> = Vec::new();
            for value in tensors.values() {
                match value {
                    Value::Array(rows) => inferred.push(rows.len()),
                    _ => return Err("ONNX input tensor must be array-like".to_string()),
                }
            }
            if inferred.is_empty() {
                return Err("Parsed input contained no features/tensors".to_string());
            }
            if inferred.contains(&0) {
                return Err("ONNX_DYNAMIC_BATCH enabled but batch dimension invalid".to_string());
            }
            if inferred.windows(2).any(|w| w[0] != w[1]) {
                return Err(format!(
                    "ONNX inputs have mismatched batch sizes: {inferred:?}"
                ));
            }
            return Ok(inferred[0]);
        }
        Err("Parsed input contained no features/tensors".to_string())
    }
}

pub trait BaseAdapter: Send + Sync {
    fn is_ready(&self) -> bool;
    fn predict(&self, parsed_input: &ParsedInput) -> Result<Value, String>;
}

type OnnxRunnableModel = RunnableModel<
    TypedFact,
    Box<dyn TypedOp>,
    tract_onnx::prelude::Graph<TypedFact, Box<dyn TypedOp>>,
>;

#[derive(Clone)]
struct OnnxAdapter {
    cfg: AppConfig,
    model: Option<OnnxRunnableModel>,
    output_map: HashMap<String, String>,
}

impl OnnxAdapter {
    fn new(cfg: AppConfig) -> Result<Self, String> {
        let path = cfg.model_path();
        if !path.exists() {
            return Err(format!("ONNX model not found: {}", path.display()));
        }
        let model = tract_onnx::onnx()
            .model_for_path(&path)
            .and_then(|model| model.into_optimized())
            .and_then(|model| model.into_runnable())
            .map_err(|e| {
                format!(
                    "Failed to load or prepare ONNX model {}: {}",
                    path.display(),
                    e
                )
            })?;
        let output_map = load_json_map(&cfg.onnx_output_map_json)?;
        Ok(Self {
            cfg,
            model: Some(model),
            output_map,
        })
    }

    fn parsed_input_to_rows(parsed_input: &ParsedInput) -> Result<Vec<Vec<f64>>, String> {
        if let Some(rows) = &parsed_input.x {
            return Ok(rows.clone());
        }
        if let Some(tensors) = &parsed_input.tensors {
            let first = tensors
                .values()
                .next()
                .ok_or_else(|| "Parsed input contained no tensors".to_string())?;
            return value_to_numeric_rows(first);
        }
        Err("Parsed input contained no features/tensors".to_string())
    }

    fn rows_to_tensor(rows: &[Vec<f64>]) -> Result<tract_onnx::prelude::Tensor, String> {
        if rows.is_empty() {
            return Err("Parsed payload is empty".to_string());
        }
        let n_rows = rows.len();
        let n_cols = rows[0].len();
        if rows.iter().any(|row| row.len() != n_cols) {
            return Err("Input rows have inconsistent feature counts".to_string());
        }
        let flat = rows
            .iter()
            .flat_map(|row| row.iter().copied())
            .map(|value| value as f32)
            .collect::<Vec<f32>>();
        let arr = tract_onnx::prelude::tract_ndarray::Array2::<f32>::from_shape_vec(
            (n_rows, n_cols),
            flat,
        )
        .map_err(|err| format!("failed to build input tensor: {err}"))?;
        Ok(arr.into())
    }

    fn tensor_to_json(tensor: &tract_onnx::prelude::Tensor) -> Result<Value, String> {
        if let Ok(view) = tensor.to_array_view::<f32>() {
            if view.ndim() == 1 {
                let values = view
                    .iter()
                    .map(|v| Value::from(*v as f64))
                    .collect::<Vec<Value>>();
                return Ok(Value::Array(values));
            }
            if view.ndim() == 2 {
                let mut rows = Vec::new();
                for row in view.outer_iter() {
                    rows.push(Value::Array(
                        row.iter()
                            .map(|v| Value::from(*v as f64))
                            .collect::<Vec<Value>>(),
                    ));
                }
                return Ok(Value::Array(rows));
            }
            return Ok(Value::Array(
                view.iter()
                    .map(|v| Value::from(*v as f64))
                    .collect::<Vec<Value>>(),
            ));
        }
        if let Ok(view) = tensor.to_array_view::<i64>() {
            if view.ndim() == 1 {
                let values = view.iter().map(|v| Value::from(*v)).collect::<Vec<Value>>();
                return Ok(Value::Array(values));
            }
            if view.ndim() == 2 {
                let mut rows = Vec::new();
                for row in view.outer_iter() {
                    rows.push(Value::Array(
                        row.iter().map(|v| Value::from(*v)).collect::<Vec<Value>>(),
                    ));
                }
                return Ok(Value::Array(rows));
            }
            return Ok(Value::Array(
                view.iter().map(|v| Value::from(*v)).collect::<Vec<Value>>(),
            ));
        }
        Err("unsupported ONNX output tensor dtype".to_string())
    }
}

impl BaseAdapter for OnnxAdapter {
    fn is_ready(&self) -> bool {
        self.model.is_some()
    }

    fn predict(&self, parsed_input: &ParsedInput) -> Result<Value, String> {
        let rows = Self::parsed_input_to_rows(parsed_input)?;
        let input = Self::rows_to_tensor(&rows)?;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| "ONNX model runtime unavailable".to_string())?;
        let outputs = model
            .run(tvec!(input.into()))
            .map_err(|err| format!("ONNX inference failed: {err}"))?;

        if !self.output_map.is_empty() {
            let mut mapped = serde_json::Map::new();
            for (response_key, onnx_output_name) in &self.output_map {
                let index = onnx_output_name
                    .parse::<usize>()
                    .unwrap_or(0)
                    .min(outputs.len().saturating_sub(1));
                mapped.insert(response_key.clone(), Self::tensor_to_json(&outputs[index])?);
            }
            return Ok(Value::Object(mapped));
        }

        if !self.cfg.onnx_output_name.trim().is_empty() {
            let index = self
                .cfg
                .onnx_output_name
                .parse::<usize>()
                .unwrap_or(self.cfg.onnx_output_index)
                .min(outputs.len().saturating_sub(1));
            return Self::tensor_to_json(&outputs[index]);
        }

        let index = self
            .cfg
            .onnx_output_index
            .min(outputs.len().saturating_sub(1));
        Self::tensor_to_json(&outputs[index])
    }
}

fn load_adapter(cfg: &AppConfig) -> Result<Arc<dyn BaseAdapter>, String> {
    let path = cfg.model_path();
    if cfg.model_type == "onnx" || path.exists() {
        let adapter = OnnxAdapter::new(cfg.clone())?;
        return Ok(Arc::new(adapter));
    }
    if !cfg.model_type.is_empty() && cfg.model_type != "onnx" {
        return Err(format!(
            "MODEL_TYPE={} is not implemented in this package",
            cfg.model_type
        ));
    }
    Err("Set MODEL_TYPE=onnx or place model.onnx under SM_MODEL_DIR".to_string())
}

#[derive(Clone)]
pub struct AppState {
    cfg: AppConfig,
    adapter: Arc<RwLock<Option<Arc<dyn BaseAdapter>>>>,
    inflight: Arc<Semaphore>,
}

impl AppState {
    pub fn new(cfg: AppConfig) -> Self {
        let max_inflight = cfg.max_inflight.max(1);
        Self {
            cfg,
            adapter: Arc::new(RwLock::new(None)),
            inflight: Arc::new(Semaphore::new(max_inflight)),
        }
    }

    async fn ensure_adapter_loaded(&self) -> Result<Arc<dyn BaseAdapter>, String> {
        if let Some(existing) = self.adapter.read().await.as_ref() {
            return Ok(existing.clone());
        }
        let loaded = load_adapter(&self.cfg)?;
        let mut writer = self.adapter.write().await;
        *writer = Some(loaded.clone());
        Ok(loaded)
    }

    fn parse_payload(&self, payload: &[u8], content_type: &str) -> Result<ParsedInput, String> {
        if self.cfg.input_mode != "tabular" {
            return Err(format!(
                "INPUT_MODE={} not implemented (tabular only for now)",
                self.cfg.input_mode
            ));
        }

        let normalized = strip_content_type_params(content_type);
        let onnx_input_map = load_json_map(&self.cfg.onnx_input_map_json)?;
        if !onnx_input_map.is_empty() && is_json_content_type(&normalized) {
            return self.parse_onnx_multi_input(payload, &normalized, &onnx_input_map);
        }

        let mut matrix = if CSV_CONTENT_TYPES.contains(&normalized.as_str()) {
            parse_csv_rows(payload, &self.cfg)?
        } else if JSON_CONTENT_TYPES.contains(&normalized.as_str()) {
            parse_json_rows(payload, &self.cfg)?
        } else if JSON_LINES_CONTENT_TYPES.contains(&normalized.as_str()) {
            parse_jsonl_rows(payload, &self.cfg)?
        } else {
            return Err(format!("Unsupported Content-Type: {content_type}"));
        };

        if matrix.is_empty() {
            return Err("Parsed payload is empty".to_string());
        }
        if self.cfg.tabular_num_features > 0 {
            let got = matrix.first().map_or(0, |r| r.len());
            if got != self.cfg.tabular_num_features {
                return Err(format!(
                    "Feature count mismatch: got {got} expected TABULAR_NUM_FEATURES={}",
                    self.cfg.tabular_num_features
                ));
            }
        }

        // Keep behavior-compatible hook for id/feature selectors without materializing
        // split outputs yet; this preserves schema knobs at config level.
        if !self.cfg.tabular_feature_columns.is_empty() || !self.cfg.tabular_id_columns.is_empty() {
            let feature_idx = if !self.cfg.tabular_feature_columns.is_empty() {
                parse_col_selector(
                    &self.cfg.tabular_feature_columns,
                    matrix.first().map_or(0, |r| r.len()),
                )?
            } else {
                let n_cols = matrix.first().map_or(0, |r| r.len());
                let id_idx = parse_col_selector(&self.cfg.tabular_id_columns, n_cols)?;
                (0..n_cols)
                    .filter(|col| !id_idx.contains(col))
                    .collect::<Vec<usize>>()
            };
            matrix = matrix
                .iter()
                .map(|row| {
                    feature_idx
                        .iter()
                        .map(|idx| row[*idx])
                        .collect::<Vec<f64>>()
                })
                .collect::<Vec<Vec<f64>>>();
        }

        Ok(ParsedInput {
            x: Some(matrix),
            tensors: None,
            meta: None,
        })
    }

    fn parse_onnx_multi_input(
        &self,
        payload: &[u8],
        content_type: &str,
        onnx_input_map: &HashMap<String, String>,
    ) -> Result<ParsedInput, String> {
        let records = if JSON_CONTENT_TYPES.contains(&content_type) {
            parse_json_records(payload, &self.cfg)?
        } else {
            parse_jsonl_records(payload)?
        };

        if records.is_empty() {
            return Err(
                "ONNX multi-input mode expects a JSON object or a non-empty list of objects"
                    .to_string(),
            );
        }
        let dtype_map = load_json_map(&self.cfg.onnx_input_dtype_map_json)?;
        let mut tensors = HashMap::new();
        let mut batch_sizes = Vec::new();

        for (request_key, onnx_input_name) in onnx_input_map {
            let mut values: Vec<Value> = Vec::new();
            for record in &records {
                let value = record.get(request_key).ok_or_else(|| {
                    format!(
                        "Missing key '{}' in one of the records for ONNX multi-input",
                        request_key
                    )
                })?;
                values.push(value.clone());
            }
            let _dtype_hint = dtype_map
                .get(request_key)
                .or_else(|| dtype_map.get(onnx_input_name))
                .cloned()
                .unwrap_or_else(|| self.cfg.tabular_dtype.clone());
            if self.cfg.onnx_dynamic_batch {
                batch_sizes.push(values.len());
            }
            tensors.insert(onnx_input_name.clone(), Value::Array(values));
        }

        if self.cfg.onnx_dynamic_batch {
            if batch_sizes.is_empty() || batch_sizes.contains(&0) {
                return Err("ONNX_DYNAMIC_BATCH enabled but batch dimension invalid".to_string());
            }
            if batch_sizes.windows(2).any(|w| w[0] != w[1]) {
                return Err(format!(
                    "ONNX inputs have mismatched batch sizes: {batch_sizes:?}"
                ));
            }
        }

        Ok(ParsedInput {
            x: None,
            tensors: Some(tensors),
            meta: Some(json!({"records": records.len(), "mode": "onnx_multi_input"})),
        })
    }

    fn format_output(&self, predictions: Value, accept: &str) -> Result<(Vec<u8>, String), String> {
        if predictions.is_object() {
            let bytes = serde_json::to_vec(&predictions)
                .map_err(|err| format!("failed to encode json: {err}"))?;
            return Ok((bytes, "application/json".to_string()));
        }

        let normalized_accept = accept
            .split(',')
            .next()
            .unwrap_or(self.cfg.default_accept.as_str())
            .trim()
            .to_ascii_lowercase();
        if CSV_CONTENT_TYPES.contains(&normalized_accept.as_str()) {
            let csv = format_csv_predictions(&predictions, &self.cfg.csv_delimiter)?;
            return Ok((csv.into_bytes(), "text/csv".to_string()));
        }

        let payload = if self.cfg.predictions_only {
            predictions
        } else {
            let mut output: serde_json::Map<String, Value> = Default::default();
            output.insert(self.cfg.json_output_key.clone(), predictions);
            Value::Object(output)
        };
        let bytes =
            serde_json::to_vec(&payload).map_err(|err| format!("failed to encode json: {err}"))?;
        Ok((bytes, "application/json".to_string()))
    }
}

fn strip_content_type_params(content_type: &str) -> String {
    content_type
        .split(';')
        .next()
        .unwrap_or(content_type)
        .trim()
        .to_ascii_lowercase()
}

fn is_json_content_type(content_type: &str) -> bool {
    JSON_CONTENT_TYPES.contains(&content_type) || JSON_LINES_CONTENT_TYPES.contains(&content_type)
}

fn load_json_map(raw: &str) -> Result<HashMap<String, String>, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(HashMap::new());
    }
    let value: Value = serde_json::from_str(trimmed)
        .map_err(|err| format!("Expected JSON object mapping: {err}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "Expected JSON object mapping".to_string())?;
    let mut out = HashMap::new();
    for (key, val) in object {
        out.insert(key.clone(), val.as_str().unwrap_or("").to_string());
    }
    Ok(out)
}

fn parse_json_records(
    payload: &[u8],
    cfg: &AppConfig,
) -> Result<Vec<HashMap<String, Value>>, String> {
    let value: Value =
        serde_json::from_slice(payload).map_err(|err| format!("invalid json payload: {err}"))?;
    let scoped = if value.is_object() {
        if let Some(field) = value.get(&cfg.json_key_instances) {
            field.clone()
        } else {
            value
        }
    } else {
        value
    };
    if let Some(obj) = scoped.as_object() {
        return Ok(vec![obj
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<HashMap<String, Value>>()]);
    }
    let arr = scoped.as_array().ok_or_else(|| {
        "ONNX multi-input mode expects a JSON object or a non-empty list of objects".to_string()
    })?;
    let mut out = Vec::new();
    for item in arr {
        let map = item.as_object().ok_or_else(|| {
            "ONNX multi-input mode expects each record to be a JSON object".to_string()
        })?;
        out.push(
            map.iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect::<HashMap<String, Value>>(),
        );
    }
    Ok(out)
}

fn parse_jsonl_records(payload: &[u8]) -> Result<Vec<HashMap<String, Value>>, String> {
    let text =
        std::str::from_utf8(payload).map_err(|err| format!("invalid utf-8 payload: {err}"))?;
    let mut out = Vec::new();
    for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let value: Value = serde_json::from_str(line)
            .map_err(|err| format!("invalid json line payload: {err}"))?;
        let map = value.as_object().ok_or_else(|| {
            "ONNX multi-input mode expects each record to be a JSON object".to_string()
        })?;
        out.push(
            map.iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect::<HashMap<String, Value>>(),
        );
    }
    Ok(out)
}

fn parse_json_rows(payload: &[u8], cfg: &AppConfig) -> Result<Vec<Vec<f64>>, String> {
    let value: Value =
        serde_json::from_slice(payload).map_err(|err| format!("invalid json payload: {err}"))?;
    let scoped = if let Some(instances) = value.get(&cfg.json_key_instances) {
        instances.clone()
    } else if let Some(features) = value.get(&cfg.jsonl_features_key) {
        Value::Array(vec![features.clone()])
    } else {
        value
    };
    value_to_numeric_rows(&scoped)
}

fn parse_jsonl_rows(payload: &[u8], cfg: &AppConfig) -> Result<Vec<Vec<f64>>, String> {
    let text =
        std::str::from_utf8(payload).map_err(|err| format!("invalid utf-8 payload: {err}"))?;
    let mut rows = Vec::new();
    for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let value: Value = serde_json::from_str(line)
            .map_err(|err| format!("invalid json line payload: {err}"))?;
        if let Some(obj) = value.as_object() {
            if let Some(features) = obj.get(&cfg.jsonl_features_key) {
                rows.extend(value_to_numeric_rows(features)?);
                continue;
            }
        }
        rows.extend(value_to_numeric_rows(&value)?);
    }
    Ok(rows)
}

fn value_to_numeric_rows(value: &Value) -> Result<Vec<Vec<f64>>, String> {
    if let Some(arr) = value.as_array() {
        if arr.first().is_some_and(|item| item.is_array()) {
            return arr
                .iter()
                .map(|row| {
                    row.as_array()
                        .ok_or_else(|| "Expected array row".to_string())?
                        .iter()
                        .map(|item| {
                            item.as_f64()
                                .ok_or_else(|| "Expected numeric value in payload".to_string())
                        })
                        .collect::<Result<Vec<f64>, String>>()
                })
                .collect::<Result<Vec<Vec<f64>>, String>>();
        }
        return Ok(vec![arr
            .iter()
            .map(|item| {
                item.as_f64()
                    .ok_or_else(|| "Expected numeric value in payload".to_string())
            })
            .collect::<Result<Vec<f64>, String>>()?]);
    }
    if let Some(number) = value.as_f64() {
        return Ok(vec![vec![number]]);
    }
    Err("Expected tabular numeric payload".to_string())
}

fn parse_csv_rows(payload: &[u8], cfg: &AppConfig) -> Result<Vec<Vec<f64>>, String> {
    let text =
        std::str::from_utf8(payload).map_err(|err| format!("invalid utf-8 csv payload: {err}"))?;
    let mut lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !cfg.csv_skip_blank_lines || !line.is_empty())
        .collect::<Vec<&str>>();
    if lines.is_empty() {
        return Err("Empty CSV payload".to_string());
    }
    match cfg.csv_has_header.as_str() {
        "true" => {
            lines.remove(0);
        }
        "auto" => {
            if csv_first_row_is_header(lines[0], cfg.csv_delimiter.as_str()) {
                lines.remove(0);
            }
        }
        "false" => {}
        _ => return Err("CSV_HAS_HEADER must be auto|true|false".to_string()),
    }
    if lines.is_empty() {
        return Err("CSV payload contains only header row".to_string());
    }
    let delim = cfg.csv_delimiter.as_str();
    lines
        .iter()
        .map(|line| {
            line.split(delim)
                .map(|token| {
                    token
                        .trim()
                        .parse::<f64>()
                        .map_err(|_| "Expected numeric value in CSV payload".to_string())
                })
                .collect::<Result<Vec<f64>, String>>()
        })
        .collect::<Result<Vec<Vec<f64>>, String>>()
}

fn csv_first_row_is_header(line: &str, delim: &str) -> bool {
    line.split(delim)
        .any(|token| token.trim().parse::<f64>().is_err())
}

fn parse_col_selector(selector: &str, n_cols: usize) -> Result<Vec<usize>, String> {
    let trimmed = selector.trim();
    if trimmed.is_empty() {
        return Ok((0..n_cols).collect::<Vec<usize>>());
    }
    if let Some((start_raw, end_raw)) = trimmed.split_once(':') {
        let start = if start_raw.is_empty() {
            0
        } else {
            start_raw
                .parse::<usize>()
                .map_err(|_| "Invalid column selector".to_string())?
        };
        let end = if end_raw.is_empty() {
            n_cols
        } else {
            end_raw
                .parse::<usize>()
                .map_err(|_| "Invalid column selector".to_string())?
        };
        let bounded_end = end.min(n_cols);
        return Ok((start.min(bounded_end)..bounded_end).collect::<Vec<usize>>());
    }
    trimmed
        .split(',')
        .filter(|tok| !tok.trim().is_empty())
        .map(|tok| {
            tok.trim()
                .parse::<usize>()
                .map_err(|_| "Invalid column selector".to_string())
        })
        .collect::<Result<Vec<usize>, String>>()
}

fn format_csv_predictions(predictions: &Value, delimiter: &str) -> Result<String, String> {
    if let Some(rows) = predictions.as_array() {
        if rows.first().is_some_and(|item| item.is_array()) {
            let mut out = Vec::new();
            for row in rows {
                let cols = row
                    .as_array()
                    .ok_or_else(|| "expected csv row array".to_string())?
                    .iter()
                    .map(value_to_string)
                    .collect::<Vec<String>>();
                out.push(cols.join(delimiter));
            }
            return Ok(out.join("\n"));
        }
        let lines = rows.iter().map(value_to_string).collect::<Vec<String>>();
        return Ok(lines.join("\n"));
    }
    Ok(value_to_string(predictions))
}

fn value_to_string(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(v) => v.to_string(),
        Value::Number(v) => v.to_string(),
        Value::String(v) => v.clone(),
        _ => value.to_string(),
    }
}

pub fn build_http_router(cfg: AppConfig) -> Router {
    let metrics_path = cfg.prometheus_path.clone();
    let prometheus_enabled = cfg.prometheus_enabled;
    let state = Arc::new(AppState::new(cfg));
    let mut router = Router::new()
        .route("/live", get(http_live))
        .route("/healthz", get(http_live))
        .route("/ready", get(http_ready))
        .route("/readyz", get(http_ready))
        .route("/ping", get(http_ready))
        .route("/invocations", post(http_invocations));
    if prometheus_enabled {
        router = router.route(metrics_path.as_str(), get(http_metrics));
    }
    router.with_state(state)
}

async fn http_live() -> impl IntoResponse {
    (StatusCode::OK, "\n")
}

async fn http_ready(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match state.ensure_adapter_loaded().await {
        Ok(adapter) if adapter.is_ready() => (StatusCode::OK, "\n").into_response(),
        Ok(_) => (StatusCode::INTERNAL_SERVER_ERROR, "\n").into_response(),
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, "\n").into_response(),
    }
}

async fn http_metrics() -> impl IntoResponse {
    (
        StatusCode::OK,
        "# HELP byoc_up Service readiness\n# TYPE byoc_up gauge\nbyoc_up 1\n",
    )
        .into_response()
}

async fn http_invocations(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    payload: Bytes,
) -> AxumResponse {
    if payload.len() > state.cfg.max_body_bytes {
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(json!({
                "error": "payload_too_large",
                "max_bytes": state.cfg.max_body_bytes
            })),
        )
            .into_response();
    }

    let content_type = header_value_with_fallback(
        &headers,
        CONTENT_TYPE,
        SAGEMAKER_CONTENT_TYPE_HEADER,
        state.cfg.default_content_type.as_str(),
    );
    let accept = header_value_with_fallback(
        &headers,
        ACCEPT,
        SAGEMAKER_ACCEPT_HEADER,
        state.cfg.default_accept.as_str(),
    );
    let result = {
        let _permit = match timeout(
            Duration::from_secs_f64(state.cfg.acquire_timeout_s.max(0.0)),
            state.inflight.clone().acquire_owned(),
        )
        .await
        {
            Ok(Ok(permit)) => permit,
            _ => {
                return (
                    StatusCode::TOO_MANY_REQUESTS,
                    Json(json!({"error": "too_many_requests"})),
                )
                    .into_response();
            }
        };

        let adapter = match state.ensure_adapter_loaded().await {
            Ok(adapter) => adapter,
            Err(err) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": err})),
                )
                    .into_response();
            }
        };
        let parsed = match state.parse_payload(payload.as_ref(), content_type.as_str()) {
            Ok(parsed) => parsed,
            Err(err) => {
                return (StatusCode::BAD_REQUEST, Json(json!({ "error": err }))).into_response();
            }
        };
        let batch = match parsed.batch_size() {
            Ok(size) => size,
            Err(err) => {
                return (StatusCode::BAD_REQUEST, Json(json!({ "error": err }))).into_response();
            }
        };
        if batch > state.cfg.max_records {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": format!("too_many_records: {batch} > {}", state.cfg.max_records) })),
            )
                .into_response();
        }
        let predictions = match adapter.predict(&parsed) {
            Ok(predictions) => predictions,
            Err(err) => {
                return (StatusCode::BAD_REQUEST, Json(json!({ "error": err }))).into_response();
            }
        };
        state.format_output(predictions, accept.as_str())
    };

    match result {
        Ok((body, content_type)) => {
            let mut response = (StatusCode::OK, body).into_response();
            if let Ok(header) = content_type.parse() {
                response.headers_mut().insert(CONTENT_TYPE, header);
            }
            attach_trace_correlation_headers(&mut response);
            response
        }
        Err(err) => (StatusCode::BAD_REQUEST, Json(json!({ "error": err }))).into_response(),
    }
}

fn attach_trace_correlation_headers(response: &mut AxumResponse) {
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

fn header_value_with_fallback(
    headers: &HeaderMap,
    primary: HeaderName,
    fallback: &str,
    default: &str,
) -> String {
    if let Some(value) = headers.get(primary).and_then(|h| h.to_str().ok()) {
        return value.to_string();
    }
    if let Ok(fallback_name) = HeaderName::from_lowercase(fallback.as_bytes()) {
        if let Some(value) = headers.get(fallback_name).and_then(|h| h.to_str().ok()) {
            return value.to_string();
        }
    }
    default.to_string()
}

pub async fn serve_http(listener: TcpListener, cfg: AppConfig) -> Result<(), std::io::Error> {
    let app = build_http_router(cfg);
    axum::serve(listener, app).await
}

pub async fn serve_grpc(
    listener: TcpListener,
    cfg: AppConfig,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let service = InferenceGrpcService::new(cfg);
    Server::builder()
        .add_service(grpc::inference_service_server::InferenceServiceServer::new(
            service,
        ))
        .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
        .await?;
    Ok(())
}

pub async fn run_http_server(host: &str, port: u16, cfg: AppConfig) -> Result<(), std::io::Error> {
    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .expect("valid listen address");
    let listener = TcpListener::bind(addr).await?;
    serve_http(listener, cfg).await
}

pub async fn run_grpc_server(
    host: &str,
    port: u16,
    cfg: AppConfig,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .expect("valid listen address");
    let listener = TcpListener::bind(addr).await?;
    serve_grpc(listener, cfg).await
}

#[derive(Clone)]
pub struct InferenceGrpcService {
    state: AppState,
    load_error: Option<String>,
}

impl InferenceGrpcService {
    fn new(cfg: AppConfig) -> Self {
        let state = AppState::new(cfg.clone());
        match load_adapter(&cfg) {
            Ok(adapter) => {
                // Pre-populate the adapter in the state
                let adapter_clone = adapter.clone();
                let state_clone = state.clone();
                tokio::spawn(async move {
                    let mut writer = state_clone.adapter.write().await;
                    *writer = Some(adapter_clone);
                });
                Self {
                    state,
                    load_error: None,
                }
            }
            Err(err) => Self {
                state,
                load_error: Some(err),
            },
        }
    }
}

#[tonic::async_trait]
impl grpc::inference_service_server::InferenceService for InferenceGrpcService {
    async fn live(
        &self,
        _request: Request<grpc::LiveRequest>,
    ) -> Result<Response<grpc::StatusReply>, Status> {
        Ok(Response::new(grpc::StatusReply {
            ok: true,
            status: "live".to_string(),
        }))
    }

    async fn ready(
        &self,
        _request: Request<grpc::ReadyRequest>,
    ) -> Result<Response<grpc::StatusReply>, Status> {
        let ready = self
            .state
            .adapter
            .read()
            .await
            .as_ref()
            .is_some_and(|adapter| adapter.is_ready());
        Ok(Response::new(grpc::StatusReply {
            ok: ready,
            status: if ready {
                "ready".to_string()
            } else {
                "not_ready".to_string()
            },
        }))
    }

    async fn predict(
        &self,
        request: Request<grpc::PredictRequest>,
    ) -> Result<Response<grpc::PredictReply>, Status> {
        if let Some(err) = &self.load_error {
            return Err(Status::new(Code::Internal, err.clone()));
        }
        let req = request.into_inner();

        // Enforce max_body_bytes to maintain HTTP/gRPC parity
        if req.payload.len() > self.state.cfg.max_body_bytes {
            return Err(Status::new(
                Code::InvalidArgument,
                format!(
                    "payload too large: {} bytes > {} bytes limit",
                    req.payload.len(),
                    self.state.cfg.max_body_bytes
                ),
            ));
        }

        // Apply max_inflight semaphore for HTTP/gRPC parity
        let _permit = match timeout(
            Duration::from_secs_f64(self.state.cfg.acquire_timeout_s.max(0.0)),
            self.state.inflight.clone().acquire_owned(),
        )
        .await
        {
            Ok(Ok(permit)) => permit,
            _ => {
                return Err(Status::new(
                    Code::ResourceExhausted,
                    "too_many_requests",
                ));
            }
        };

        let content_type = if req.content_type.is_empty() {
            self.state.cfg.default_content_type.as_str()
        } else {
            req.content_type.as_str()
        };
        let accept = if req.accept.is_empty() {
            self.state.cfg.default_accept.as_str()
        } else {
            req.accept.as_str()
        };

        let adapter = self
            .state
            .ensure_adapter_loaded()
            .await
            .map_err(|err| Status::new(Code::Internal, err))?;
        let parsed = self
            .state
            .parse_payload(req.payload.as_ref(), content_type)
            .map_err(|err| Status::new(Code::InvalidArgument, err))?;
        let batch = parsed
            .batch_size()
            .map_err(|err| Status::new(Code::InvalidArgument, err))?;
        if batch > self.state.cfg.max_records {
            return Err(Status::new(
                Code::InvalidArgument,
                format!("too_many_records: {batch} > {}", self.state.cfg.max_records),
            ));
        }
        let predictions = adapter
            .predict(&parsed)
            .map_err(|err| Status::new(Code::InvalidArgument, err))?;
        let (body, output_content_type) = self
            .state
            .format_output(predictions, accept)
            .map_err(|err| Status::new(Code::InvalidArgument, err))?;
        let mut metadata = HashMap::new();
        metadata.insert("batch_size".to_string(), batch.to_string());
        Ok(Response::new(grpc::PredictReply {
            body,
            content_type: output_content_type,
            metadata,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_instances_are_parsed() {
        let cfg = AppConfig::default();
        let state = AppState::new(cfg);
        let payload = br#"{"instances":[[1.0,2.0],[3.0,4.0]]}"#;
        let parsed = state
            .parse_payload(payload, "application/json")
            .expect("json parse should pass");
        assert_eq!(parsed.batch_size().expect("batch"), 2);
    }

    #[test]
    fn csv_header_auto_detect_works() {
        let cfg = AppConfig {
            csv_has_header: "auto".to_string(),
            ..AppConfig::default()
        };
        let state = AppState::new(cfg);
        let payload = b"f1,f2\n1,2\n3,4\n";
        let parsed = state
            .parse_payload(payload, "text/csv")
            .expect("csv parse should pass");
        assert_eq!(parsed.batch_size().expect("batch"), 2);
    }

    #[test]
    fn output_formatter_supports_csv() {
        let cfg = AppConfig::default();
        let state = AppState::new(cfg);
        let (body, content_type) = state
            .format_output(
                Value::Array(vec![Value::from(1), Value::from(2)]),
                "text/csv",
            )
            .expect("csv format should pass");
        assert_eq!(content_type, "text/csv");
        assert_eq!(String::from_utf8(body).expect("utf8"), "1\n2");
    }
}
