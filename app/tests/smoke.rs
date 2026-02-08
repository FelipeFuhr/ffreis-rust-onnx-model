#[test]
fn smoke_passes() {
    // Simple passing test to ensure CI/tests succeed
    assert_eq!(app::greet(), "Hello, world!");
}

#[test]
fn add_works() {
    assert_eq!(app::add(2, 3), 5);
}

#[test]
fn is_even_works() {
    assert!(app::is_even(2));
    assert!(app::is_even(0));
    assert!(!app::is_even(3));
}

#[test]
fn clamp_works() {
    assert_eq!(app::clamp(-1, 0, 10), 0);
    assert_eq!(app::clamp(5, 0, 10), 5);
    assert_eq!(app::clamp(11, 0, 10), 10);
}

#[test]
fn repeat_word_works() {
    assert_eq!(app::repeat_word("hi", 0), "");
    assert_eq!(app::repeat_word("hi", 1), "hi");
    assert_eq!(app::repeat_word("hi", 3), "hi hi hi");
}

#[test]
fn sum_works() {
    assert_eq!(app::sum(&[]), 0);
    assert_eq!(app::sum(&[1, 2, 3]), 6);
    assert_eq!(app::sum(&[-2, 2, 5]), 5);
}

#[test]
fn first_non_empty_works() {
    assert_eq!(app::first_non_empty(&["", "a", "b"]), Some("a"));
    assert_eq!(app::first_non_empty(&["", "", "x"]), Some("x"));
    assert_eq!(app::first_non_empty(&[""]), None);
}

#[test]
fn toggle_works() {
    assert!(app::toggle(false));
    assert!(!app::toggle(true));
}

#[test]
fn binary_runs() {
    let exe = match std::env::var("CARGO_BIN_EXE_app") {
        Ok(path) => path,
        Err(_) => return, // Skip when binary is not built in this context
    };

    let output = std::process::Command::new(exe)
        .output()
        .expect("run app binary");

    assert!(output.status.success());
    assert_eq!(
        String::from_utf8_lossy(&output.stdout).trim(),
        "Hello, world!"
    );
}
