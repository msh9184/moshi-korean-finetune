# Rust core wiring for the pluggable backbone

The new modules in `moshi-core/src/` (`hf_backbone_lm.rs`, `mistral_backend.rs`,
`dimension_adapter.rs`, `streaming_lm.rs`) are drop-in files. Wiring them into the upstream
[Kyutai Moshi](https://github.com/kyutai-labs/moshi) Rust backend requires a small set of edits
to the following upstream files (kept here as a description rather than a vendored copy of
upstream source):

| Upstream file | Edit |
|---|---|
| `rust/moshi-core/src/lib.rs` | Register the new modules (`pub mod hf_backbone_lm; pub mod mistral_backend; pub mod dimension_adapter; pub mod streaming_lm;`). |
| `rust/moshi-core/src/lm.rs` | Add a `BackendType` enum (`Moshi` \| `HfBackbone`) and a `needs_dimension_adapter()` helper; branch model construction on it. |
| `rust/moshi-backend/src/standalone.rs` | Loader branch that, for `backend_type = "hf_backbone"`, loads the backbone + adapter weights alongside the Moshi components. |
| `rust/moshi-backend/src/main.rs` | Surface the `backend_type` config field. |
| `rust/moshi-backend/src/stream_both.rs` | Route generation through the streaming-LM dispatcher. |
| `rust/moshi-core/src/lm_generate_multistream.rs` | Minor hook for the streaming-LM enum. |

The default path (`backend_type` absent or `"moshi"`) is unchanged from upstream; the
`hf_backbone` path is additive.
