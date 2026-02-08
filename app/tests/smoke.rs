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
