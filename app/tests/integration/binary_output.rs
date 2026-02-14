use crate::common::run_app_binary;

#[test]
fn binary_runs_successfully() {
    let output = match run_app_binary() {
        Some(output) => output,
        None => return, // Skip when the binary is not available in this context.
    };

    assert!(output.status.success());
}

#[test]
fn binary_prints_expected_message() {
    let output = match run_app_binary() {
        Some(output) => output,
        None => return, // Skip when the binary is not available in this context.
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout.trim(), "Hello, world!");
    assert!(stdout.ends_with('\n'));
}
