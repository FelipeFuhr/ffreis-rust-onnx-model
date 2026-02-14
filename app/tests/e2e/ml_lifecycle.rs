#[test]
fn model_fixture_path_is_configurable() {
    let model_path =
        std::env::var("TEST_MODEL_PATH").unwrap_or_else(|_| "fixtures/model.onnx".to_string());
    assert!(!model_path.trim().is_empty());
}

#[test]
#[ignore = "Enable once ONNX loading/inference API is implemented"]
fn loads_model_and_runs_inference() {
    let model_path = std::env::var("TEST_MODEL_PATH")
        .expect("set TEST_MODEL_PATH to a real model before running this test");
    assert!(
        std::path::Path::new(&model_path).exists(),
        "TEST_MODEL_PATH does not exist: {model_path}"
    );

    // Future shape:
    // let engine = app::ml::Engine::load(&model_path).expect("load model");
    // let output = engine.infer(vec![0.0_f32; N]).expect("run inference");
    // assert_eq!(output.len(), EXPECTED_OUTPUT_SIZE);
}

#[test]
#[ignore = "Enable once model fine-tuning/training path is implemented"]
fn trains_model_from_fixture_data() {
    let dataset_path = std::env::var("TEST_DATASET_PATH")
        .expect("set TEST_DATASET_PATH to a training dataset before running this test");
    assert!(
        std::path::Path::new(&dataset_path).exists(),
        "TEST_DATASET_PATH does not exist: {dataset_path}"
    );

    // Future shape:
    // let trainer = app::ml::Trainer::new();
    // let result = trainer.train(&dataset_path).expect("train model");
    // assert!(result.loss < 0.10);
}
