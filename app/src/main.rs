fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod main_tests {
    use crate::add;

    #[test]
    fn sanity() {
        assert_eq!(add(1, 1), 2);
    }
}
