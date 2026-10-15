NIGHTLY_RUST_VERSION ?= nightly
TAPLO_CLI_VERSION ?= 0.9.3
CARGO_MUTANTS_VERSION ?= 27.1.0
# The base of `make mutants-diff`: mutate only what changed since it.
BASE ?= origin/main

# Crates extracted to Lean with Charon/Aeneas. They must stay inside the
# subset of Rust Aeneas supports (no unsafe, no closures or dyn, no iterator
# chains in non-test code).
EXTRACTED_CRATES ?= arrabbiata-field arrabbiata-poly arrabbiata-transcript \
	arrabbiata-commit arrabbiata-relation arrabbiata-folding

.PHONY: help
help: ## Ask for help!
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; \
		{printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: build
build: ## Build the workspace in debug mode
	cargo build --workspace --all-targets

.PHONY: build-release
build-release: ## Build the workspace in release mode
	cargo build --workspace --all-targets --release

.PHONY: check
check: ## Check the workspace for compilation errors
	cargo check --workspace --all-targets

.PHONY: check-format
check-format: ## Check Rust and TOML formatting
	cargo +$(NIGHTLY_RUST_VERSION) fmt --all -- --check
	taplo format --check

.PHONY: format
format: ## Format Rust and TOML files
	cargo +$(NIGHTLY_RUST_VERSION) fmt --all
	taplo format

.PHONY: lint
lint: ## Run clippy, denying warnings
	cargo clippy --workspace --all-targets -- -D warnings

.PHONY: lint-beta
lint-beta: ## Run clippy with the beta toolchain
	cargo +beta clippy --workspace --all-targets -- -D warnings

.PHONY: test
test: ## Run all tests, including doctests and compile-fail examples
	cargo test --workspace

.PHONY: test-curve
test-curve: ## Curve laws on CURVE (pallas, vesta, bn254_g{1,2}, grumpkin)
	cargo test -p arrabbiata-curve --test laws -- --exact $(CURVE)

.PHONY: test-release
test-release: ## Run all tests in release mode
	cargo test --workspace --release

.PHONY: doc
doc: ## Build the API documentation, denying rustdoc warnings
	RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps

.PHONY: doc-site
doc-site: ## Build the documentation site (all crates, with an index page)
	RUSTDOCFLAGS="--enable-index-page -Z unstable-options -D warnings" \
		cargo +$(NIGHTLY_RUST_VERSION) doc --workspace --no-deps

.PHONY: mutants
mutants: ## Mutation testing (optional: PACKAGE=crate, SHARD=k/n, FILE=path)
	cargo mutants --no-shuffle \
		$(if $(PACKAGE),--package $(PACKAGE)) \
		$(if $(SHARD),--shard $(SHARD)) \
		$(if $(FILE),--file $(FILE))

.PHONY: mutants-diff
mutants-diff: ## Mutation testing of the code changed since BASE
	git diff $(BASE) -- '*.rs' > mutants.diff
	cargo mutants --no-shuffle --in-diff mutants.diff

.PHONY: deny
deny: ## Check licenses, advisories, bans and sources
	cargo deny check

.PHONY: lint-shell
lint-shell: ## Lint shell scripts
	@files=$$(git ls-files '*.sh'); \
	if [ -n "$$files" ]; then shellcheck $$files; fi

.PHONY: lean-setup
lean-setup: ## Fetch Mathlib and its prebuilt cache
	cd lean && lake update && lake exe cache get

.PHONY: lean-build
lean-build: ## Build the Lean library, rejecting sorry, and check axioms
	cd lean && lake build --wfail
	./scripts/check-axioms.sh

.PHONY: extract
extract: ## Extract the core crates to Lean (needs charon and aeneas)
	@command -v charon >/dev/null || \
		{ echo "charon not found; see lean/README.md"; exit 1; }
	@command -v aeneas >/dev/null || \
		{ echo "aeneas not found; see lean/README.md"; exit 1; }
	./scripts/extract.sh $(EXTRACTED_CRATES)

.PHONY: setup-taplo
setup-taplo: ## Install the taplo TOML formatter
	@if taplo --version 2>/dev/null | grep -q $(TAPLO_CLI_VERSION); then \
		echo "taplo $(TAPLO_CLI_VERSION) already installed"; \
	else \
		cargo +$(NIGHTLY_RUST_VERSION) install taplo-cli \
			--version $(TAPLO_CLI_VERSION) --force; \
	fi

.PHONY: setup-mutants
setup-mutants: ## Install cargo-mutants
	@if cargo mutants --version 2>/dev/null | \
		grep -q $(CARGO_MUTANTS_VERSION); then \
		echo "cargo-mutants $(CARGO_MUTANTS_VERSION) already installed"; \
	else \
		cargo install cargo-mutants --version $(CARGO_MUTANTS_VERSION) \
			--locked; \
	fi

.PHONY: clean
clean: ## Clean build artifacts
	cargo clean
	rm -rf lean/.lake

.PHONY: setup
setup: setup-taplo ## Set up the development environment
	rustup toolchain install $(NIGHTLY_RUST_VERSION) --component rustfmt
	rustup toolchain install beta --component clippy
