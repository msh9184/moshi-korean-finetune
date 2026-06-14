# K-Moshi Serving — Real-Time Rust Backend & Demo

This directory completes the **train → export → serve → demo** story for the Korean Moshi
fine-tune: it serves a model trained with [`moshi-korean-finetune`](../) on the
**[Kyutai Moshi](https://github.com/kyutai-labs/moshi) Rust backend**, and exposes it to a
web client for live full-duplex Korean voice conversation.

> This is an **overlay on upstream Kyutai Moshi**, not a fork. You clone `kyutai-labs/moshi`,
> drop in the files here (and apply the small Rust patch), then run. Model paths are
> `/path/to/...` placeholders. No proprietary models/data are included.

## End-to-end pipeline

```
[ train ]   moshi-korean-finetune  ──►  LoRA adapter (lora.safetensors)
                                         on base kyutai/moshiko + Korean tokenizer + Mimi codec
                                              │
[ export ]  scripts/import_rust_lora.py  ──►  fuse LoRA into base, re-serialize for Rust/Candle
                                              │   → korean-moshi-fused.safetensors
                                              ▼
[ serve ]   scripts/serve_korean_moshi.sh ──► writes configs/config-korean.json,
                                              cargo run --release -- --config ... standalone
                                              │   (Kyutai Moshi Rust backend, HTTPS/WebSocket :8998)
                                              ▼
[ demo ]    web client (static_dir) ────────► browser: live Korean full-duplex voice chat
```

## What's in this overlay

| Path | Purpose |
|---|---|
| `scripts/import_rust_lora.py` | Export bridge: fuse the LoRA adapter into the base Moshi weights and emit a single Rust/Candle-loadable `safetensors`. |
| `scripts/serve_korean_moshi.sh` | One-command recipe: validate inputs → fuse LoRA → write config → `cargo run --release ... standalone`. |
| `configs/config-korean.json` | Rust backend config for the fine-tuned Korean Moshi (fused weights, Korean tokenizer, Mimi 8 codebooks). |
| `configs/config-hf-backbone.json` | Config for the optional pluggable backbone variant (separate backbone + dimension adapter). |
| `rust-overlay/moshi-core/src/*.rs` | **New** Rust modules adding a pluggable LM backbone to the serving core (see below). |
| `rust-overlay/PATCH.md` | The small set of edits to upstream Rust files needed to wire the pluggable backbone. |

## Pluggable Rust backbone (optional)

Beyond serving Moshi's native LM, this overlay adds a **pluggable backbone** to the Rust core so a
separate **Mistral-architecture HF model** (e.g. a 3072-dim causal LM) can replace Moshi's transformer,
bridged to Moshi's 4096-dim depformer by a learned **dimension adapter** — the Rust mirror of the
`hf_lm` backbone in the training repo.

| File | Role |
|---|---|
| `rust-overlay/moshi-core/src/hf_backbone_lm.rs` | LM wrapper exposing the alternative backbone behind the Moshi LM interface. |
| `rust-overlay/moshi-core/src/mistral_backend.rs` | Mistral-architecture backbone (GQA, sliding-window) for the Rust/Candle runtime. |
| `rust-overlay/moshi-core/src/dimension_adapter.rs` | Projects backbone hidden size ↔ Moshi's embedding/depformer dimension. |
| `rust-overlay/moshi-core/src/streaming_lm.rs` | Streaming-LM enum dispatching native Moshi vs the pluggable backbone. |

## Usage

```bash
# 1) Clone the upstream Moshi serving stack
git clone https://github.com/kyutai-labs/moshi && cd moshi

# 2) Apply this overlay (see rust-overlay/PATCH.md for the core edits)
cp -r <this-repo>/serving/rust-overlay/moshi-core/src/*.rs rust/moshi-core/src/
#    then apply the wiring edits described in rust-overlay/PATCH.md

# 3) Fuse the LoRA adapter from training into a Rust-loadable checkpoint, then serve
bash <this-repo>/serving/scripts/serve_korean_moshi.sh /path/to/lora.safetensors
#    (edit the BASE_MODEL / MIMI_MODEL / TOKENIZER placeholders at the top first)

# 4) Open the served web client in the browser (https://localhost:8998)
```

## Web client

The live demo uses a **customized, Korean-localized web client** built over the upstream Moshi client:
a WebGL/Three.js voice-orb scene, Siri-style real-time audio visualizers, a Korean conversation UI,
and mute / settings-preset / server-audio-stats panels. The client is a UI overlay of the upstream
`client/` (debranded to a neutral **K-Moshi** identity) and is described here rather than vendored;
build it from the upstream client with these components dropped in.

## Notes

- Depends on upstream **kyutai-labs/moshi** (Apache-2.0) for the Rust backend, the `moshi` Python
  package, and the Mimi codec — this overlay only carries the user-authored serving additions.
- All weights/paths are placeholders; bring your own fine-tuned checkpoint from [`../`](../).
