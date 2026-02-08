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
