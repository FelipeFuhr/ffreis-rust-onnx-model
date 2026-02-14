# Test Layout

This crate uses layered tests so it can scale to an API that loads ML models and runs inference/training.

## Unit tests

- Location: `app/src/lib.rs` (`#[cfg(test)] mod tests`)
- Scope: pure function behavior and edge cases
- Command: `make -C app test-unit`

## Integration tests

- Entrypoint: `app/tests/integration_tests.rs`
- Modules:
  - `app/tests/integration/library_api.rs`
  - `app/tests/integration/binary_output.rs`
- Shared helpers: `app/tests/common/mod.rs`
- Scope: public API contracts and binary behavior (black-box style)
- Command: `make -C app test-integration`

## E2E / ML scenario tests

- Entrypoint: `app/tests/e2e_tests.rs`
- Modules:
  - `app/tests/e2e/ml_lifecycle.rs`
- Scope: environment-driven, heavier scenarios (real model files, datasets, training runs)
- Command: `make -C app test-e2e`
- Heavy tests are marked `#[ignore]` until the ML API exists and fixtures are available.
- To run ignored tests intentionally:
  - `make -C app test-e2e-ignored`

## Full suite

- `make -C app test`
