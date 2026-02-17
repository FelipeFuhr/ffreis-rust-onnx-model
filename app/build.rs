fn main() {
    let protoc_path =
        protoc_bin_vendored::protoc_bin_path().expect("protoc binary should be available");
    // SAFETY: build script sets process env before invoking tonic-build.
    unsafe {
        std::env::set_var("PROTOC", protoc_path);
    }

    println!("cargo:rerun-if-changed=proto/onnx_serving_grpc/inference.proto");
    println!("cargo:rerun-if-env-changed=PROTOC");

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&["proto/onnx_serving_grpc/inference.proto"], &["proto"])
        .expect("gRPC protobuf compilation failed");
}
