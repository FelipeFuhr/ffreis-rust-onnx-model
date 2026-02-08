#[test]
fn smoke_passes() {
    // Simple passing test to ensure CI/tests succeed
    assert_eq!(app::greet(), "Hello, world!");
}

#[test]
fn add_works() {
    assert_eq!(app::add(2, 3), 5);
}
