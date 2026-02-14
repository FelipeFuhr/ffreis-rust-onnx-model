#[test]
fn hello_world_example_stays_the_same() {
    assert_eq!(app::greet(), "Hello, world!");
}

#[test]
fn library_functions_can_be_composed() {
    let values = [app::add(1, 2), app::add(10, -2), app::clamp(50, 0, 10)];
    assert_eq!(app::sum(&values), 21);
    assert_eq!(
        app::first_non_empty(&["", "model-ready"]),
        Some("model-ready")
    );
}
