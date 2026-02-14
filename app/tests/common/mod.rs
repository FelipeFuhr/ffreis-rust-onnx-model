use std::process::Output;

pub fn run_app_binary() -> Option<Output> {
    let exe = option_env!("CARGO_BIN_EXE_app")?;
    Some(
        std::process::Command::new(exe)
            .output()
            .expect("run app binary"),
    )
}
