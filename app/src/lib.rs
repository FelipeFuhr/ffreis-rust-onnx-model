pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub fn greet() -> &'static str {
    "Hello, world!"
}

pub fn is_even(n: i32) -> bool {
    n % 2 == 0
}

pub fn clamp(n: i32, min: i32, max: i32) -> i32 {
    if n < min {
        min
    } else if n > max {
        max
    } else {
        n
    }
}

pub fn repeat_word(word: &str, times: usize) -> String {
    std::iter::repeat_n(word, times)
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn sum(nums: &[i32]) -> i32 {
    nums.iter().copied().sum()
}

pub fn first_non_empty<'a>(values: &'a [&'a str]) -> Option<&'a str> {
    values.iter().copied().find(|v| !v.is_empty())
}

pub fn toggle(flag: bool) -> bool {
    !flag
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_handles_positive_and_negative_values() {
        assert_eq!(add(2, 3), 5);
        assert_eq!(add(-2, 3), 1);
        assert_eq!(add(-5, -7), -12);
    }

    #[test]
    fn greet_is_stable() {
        assert_eq!(greet(), "Hello, world!");
    }

    #[test]
    fn is_even_supports_negative_numbers() {
        assert!(is_even(-4));
        assert!(is_even(0));
        assert!(!is_even(-3));
    }

    #[test]
    fn clamp_limits_values_to_interval() {
        assert_eq!(clamp(-1, 0, 10), 0);
        assert_eq!(clamp(5, 0, 10), 5);
        assert_eq!(clamp(11, 0, 10), 10);
    }

    #[test]
    fn repeat_word_has_consistent_spacing() {
        assert_eq!(repeat_word("hi", 0), "");
        assert_eq!(repeat_word("hi", 1), "hi");
        assert_eq!(repeat_word("hi", 3), "hi hi hi");
    }

    #[test]
    fn sum_handles_empty_and_non_empty_lists() {
        assert_eq!(sum(&[]), 0);
        assert_eq!(sum(&[1, 2, 3]), 6);
        assert_eq!(sum(&[-2, 2, 5]), 5);
    }

    #[test]
    fn first_non_empty_returns_first_match_or_none() {
        assert_eq!(first_non_empty(&["", "a", "b"]), Some("a"));
        assert_eq!(first_non_empty(&["", "", "x"]), Some("x"));
        assert_eq!(first_non_empty(&[""]), None);
        assert_eq!(first_non_empty(&[]), None);
    }

    #[test]
    fn toggle_flips_boolean_value() {
        assert!(toggle(false));
        assert!(!toggle(true));
    }
}
