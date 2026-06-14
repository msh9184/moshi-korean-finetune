# 🇰🇷 moshi-korean-finetune

**Korean full-duplex spoken-dialogue fine-tuning on [Kyutai Moshi](https://github.com/kyutai-labs/moshi).**
A research recipe that adapts Moshi's real-time, full-duplex architecture to Korean — with a pluggable LM backbone, zero-shot speaker conditioning, Korean tokenizer tooling, a two-stage data pipeline, and an advanced monitoring/eval suite. A companion [`serving/`](serving/) overlay takes the trained model all the way to a live browser demo.

<p>
<img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-blue">
<img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue">
<img alt="base" src="https://img.shields.io/badge/base-Kyutai%20Moshi-ff5a5f">
<img alt="status" src="https://img.shields.io/badge/status-research%20PoC-orange">
</p>

> Personal research / proof-of-concept. Built on [kyutai-labs/moshi-finetune](https://github.com/kyutai-labs/moshi-finetune), inspired by [J-Moshi](https://arxiv.org/abs/2506.02979). All base models/datasets are public (Kyutai Moshi/Mimi on Hugging Face; AI-Hub / NIKL Korean corpora). Config paths are `/path/to/...` placeholders. No proprietary models or data are included.

---

## Why Korean Moshi?

Moshi models a conversation as **two parallel audio streams** plus an inner **"monologue" text stream**, enabling natural full-duplex turn-taking (overlap, backchannel, interruption). This repo explores what it takes to bring that to Korean: tokenization, Korean dialogue data (real + synthetic), speaker conditioning for voice control, and training recipes that fit a single node or scale out with FSDP/DDP.

## ✨ Key features

- **Pluggable LM backbone** — keep Moshi's native LM, or swap in any **Hugging Face causal LM** via an Abstract-Factory backbone layer + automatic **dimension adapter** bridging the LM to Moshi's depformer.
- **Zero-shot speaker conditioning** — speaker-encoder (ECAPA-TDNN / W2v-BERT 2.0 SV) + conditioner + VALL-E-style audio prompting, with unit tests.
- **Korean tokenizer tooling** — initialize/extend Moshi for Korean, SentencePiece/BBPE utilities.
- **Two-stage data pipeline** — `data_preparation/` (real dialogue → Lhotse shards) + `data_preparation_stage2/` (synthetic dialogue, partial/experimental).
- **Advanced monitoring & evaluation** — alignment / audio-quality / dialogue / semantic monitors, sample savers, BLEU, enhanced metrics (13 modules).
- **Staged training** — stage-1 pretraining + stage-2 speaker conditioning, FSDP/DDP, with a cosine-warmup scheduler and a stage-aware pretrained loader.
- **End-to-end serving** — [`serving/`](serving/): fuse LoRA → Rust backend → live Korean voice demo.

## 🆚 What's different from upstream moshi-finetune

| Capability | Upstream | **This repo** |
|---|---|---|
| LM backbone | Moshi LM only | ✅ pluggable (Moshi or any HF causal LM) + dimension adapter |
| Speaker conditioning | — | ✅ zero-shot encoder/conditioner/audio-prompt + tests |
| Korean tokenizer | — | ✅ init/extend + SentencePiece/BBPE tooling |
| Data pipeline | minimal loader | ✅ 2-stage (real Lhotse-shar + synthetic) |
| Monitoring/eval | basic logging | ✅ 13-module suite (alignment/audio/dialogue/semantic) |
| Scheduler / staged load | inline | ✅ cosine-warmup + stage-aware pretrained loader |
| Serving | — | ✅ `serving/` Rust backend overlay + LoRA-fuse bridge |

## 🏗️ Architecture (pluggable backbone)

```
            ┌──────────────────────── LMModelWrapper ────────────────────────┐
            │                                                                 │
  text/audio│   ┌───────────────── AbstractBackbone ─────────────────┐       │
  tokens  ──┼──▶│  MoshiBackbone (4096)   |   HFLMBackbone (e.g. 3072) │      │
            │   └───────────────────────────────┬─────────────────────┘      │
            │                                    │ DimensionAdapter (↔4096)   │
            │                                    ▼                            │
            │                    shared Moshi temporal transformer + depformer │
            │                                    │                            │
            └────────────────────────────────────┼────────────────────────────┘
                                                 ▼
                                  Mimi codec (audio tokens, full-duplex)
```

## 📁 Repository layout

```
finetune/
  backbone/      # pluggable LM backbone: base / factory / adapters / config
                 #   moshi_backbone.py + hf_lm_backbone.py
  modules/       # speaker conditioning: speaker_encoder / conditioner / audio_prompt
  monitoring/    # alignment / audio-quality / dialogue / semantic monitors + loggers
  data/          # dataset + interleaver (inner-monologue + dual audio streams)
  pretrained_loader.py, scheduler.py, eval.py, loss.py, args.py, ...
data_preparation/         # stage-1: real Korean dialogue → Lhotse shards
data_preparation_stage2/  # stage-2: synthetic dialogue (partial)
example/                  # korean_*.yaml training/eval configs (stages, FSDP/DDP)
tools/                    # Korean tokenizer init / wrappers / conversion
tests/                    # speaker-conditioning unit tests
serving/                  # Rust serving overlay + LoRA-fuse bridge + live demo
docs/                     # architecture / speaker-conditioning / tokenizer / recipe notes
train.py, annotate.py
```

## ⚙️ Installation

```bash
pip install -e .
# REQUIRED: install moshi / sphn / sentencepiece (pulled in --no-deps on a GPU box)
bash scripts/setup_environment.sh      # uses the public PyTorch index
```

> `pip install -e .` alone is not enough to `import moshi`; run `scripts/setup_environment.sh` (or install `moshi` manually). Base Moshi/Mimi weights come from the Kyutai Hugging Face repos.

## 🚀 Quick start

```bash
# 1) Initialize a Korean-extended Moshi checkpoint
python -m tools.init_korean_moshi --save_dir ./models/k-moshi-init --extend_modules_for_user_stream

# 2) Stage-1 data prep: real Korean dialogue → Lhotse shards
python -m data_preparation.scripts.run_phase1 --parallel
python -m data_preparation.scripts.run_phase2 --help

# 3) Stage-1 pretraining
python train.py example/korean_moshi_stage1_pretrain.yaml

# 4) Stage-2 speaker conditioning (loads stage-1 weights)
python train.py example/korean_moshi_stage2_speaker.yaml

# Multi-GPU FSDP
torchrun --nproc_per_node=4 train.py example/korean_v4_fsdp.yaml

# Speaker-conditioning evaluation
python train.py example/korean_eval_speaker_cond.yaml
```

## 🧩 Backbone selection

- **`moshi`** (default, validated) — Moshi's native LM.
- **`hf_lm`** (experimental, Phase 2) — any Hugging Face causal LM; set `hf_lm.model_path` to a checkpoint, and the dimension adapter auto-bridges mismatched widths.

```yaml
# example/korean_backbone_hf_lm.yaml
backbone:
  type: hf_lm
  hf_lm:
    model_path: /path/to/hf-causal-lm   # a Mistral-architecture model, etc.
```

## 🎚️ Speaker conditioning

Enable zero-shot voice control via `finetune/modules/`: choose a speaker encoder (`ecapa` / `w2v_bert2` / `dummy`) and an audio-prompt mode. See `docs/SPEAKER_CONDITIONING_*`.

## 🔊 Serving & live demo

[`serving/`](serving/) completes the **train → export → serve → demo** loop: fuse the LoRA adapter into a Rust/Candle-loadable checkpoint (`import_rust_lora.py`), launch the Kyutai Moshi **Rust backend** (`serve_korean_moshi.sh`), and connect a Korean web client for live full-duplex voice chat. It also adds an optional **pluggable Rust backbone** (Mistral-architecture + dimension adapter). See [`serving/README.md`](serving/README.md).

## 📈 Results

Illustrative training-loss curve (LoRA vs LoRA + embedding fine-tuning, to ~2k steps):

![Training loss curve](images/train_curve_example.png)

## 📚 Documentation

- [Architecture proposal](docs/K-MOSHI_ARCHITECTURE_PROPOSAL.md) · [Full-duplex analysis](docs/FULL_DUPLEX_ARCHITECTURE_ANALYSIS.md)
- [Speaker conditioning](docs/SPEAKER_CONDITIONING_SYSTEM_ARCHITECTURE.md) · [Zero-shot spec](docs/ZERO_SHOT_SPEAKER_CONDITIONING_SPECIFICATION.md)
- [Korean tokenizer guide](docs/KOREAN_TOKENIZER_GUIDE.md) · [Training guide](docs/TRAINING_GUIDE.md) · [Recipe analysis](docs/TRAINING_RECIPE_ANALYSIS.md)
- [Enhanced eval metrics](docs/ENHANCED_EVAL_METRICS_SPEC.md)

## 🙏 Acknowledgements

[Kyutai Moshi & moshi-finetune](https://github.com/kyutai-labs/moshi) (base architecture & framework) and [J-Moshi](https://arxiv.org/abs/2506.02979) (Japanese adaptation that inspired this Korean effort).

## 📜 License & citation

Apache-2.0 (see [LICENSE](LICENSE)), matching upstream moshi-finetune.

```bibtex
@techreport{kyutai2024moshi,
  title  = {Moshi: a speech-text foundation model for real-time dialogue},
  author = {D\'efossez, Alexandre and Mazar\'e, Laurent and Orsini, Manu and others},
  institution = {Kyutai}, year = {2024}
}
```
