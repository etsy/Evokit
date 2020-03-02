# Development

## Building

Evokit requires nightly Rust compiler.  It can be installed using `rustup`:

```bash
rustup install nightly
```

To use the nightly compiler globally, you can set it as the default:

```bash
rustup default nightly
```

If, however, you're developing multiple Rust applications, it's recommended you set it only for the given project:

```bash
cd Evokit
rustup override set nightly
```

## Installing locally
When installing locally, you will need to specify the rust flags below.
```bash
RUSTFLAGS='-C target-cpu=native' cargo install -f
```

## When submitting PRs
1. Run `cargo fix` to remove unused imports
2. Ensure `cargo test` passes
3. Add documentation
4. Run `cargo fmt`

To automatically run `rustfmt` on all your changed files, run 
```sh
cp pre-commit.sh .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
```
