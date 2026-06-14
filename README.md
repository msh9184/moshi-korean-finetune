# moshi-korean-finetune: Korean Full-Duplex Spoken-Dialogue Fine-Tuning

A research recipe for adapting **[Kyutai Moshi](https://github.com/kyutai-labs/moshi)** — a real-time, full-duplex spoken-dialogue model — to **Korean**. Built on top of [kyutai-labs/moshi-finetune](https://github.com/kyutai-labs/moshi-finetune) and inspired by [J-Moshi](https://arxiv.org/abs/2506.02979) (a Japanese Moshi adaptation), it adds a pluggable LM backbone, a zero-shot speaker-conditioning subsystem, Korean tokenizer support, a richer monitoring/evaluation suite, and a two-stage Korean data pipeline.

> Personal research / proof-of-concept. All base models and datasets referenced are public (Kyutai Moshi/Mimi on Hugging Face; AI-Hub / NIKL Korean corpora). Dataset and model paths in configs are placeholders (`/path/to/...`) for you to fill in. No proprietary models or data are included.

## Why

Moshi models a conversation as parallel audio streams plus an inner "monologue" text stream, enabling natural full-duplex turn-taking (overlap, backchannel, interruption). This repo explores what it takes to bring that architecture to Korean: tokenization, Korean dialogue data (real + synthetic), speaker conditioning for voice control, and training recipes that fit a single node or scale out with FSDP/DDP.

## Key features

- **Pluggable LM backbone** — keep Moshi's native LM, or swap in **any Hugging Face causal LM** (e.g. a Mistral-architecture model) via an Abstract-Factory backbone layer (`finetune/backbone/`), with automatic dimension adapters bridging the LM and Moshi's depformer.
- **Zero-shot speaker conditioning** — a speaker-encoder + conditioner + audio-prompt subsystem (`finetune/modules/`) to steer the generated voice, with unit tests (`tests/`).
- **Korean tokenizer support** — utilities to initialize and adapt tokenizers for Korean (`tools/`).
- **Two-stage data pipeline**
  - `data_preparation/` — turn real Korean dialogue corpora into Moshi-ready Lhotse shards (alignment, speaker selection, stereo conversion).
  - `data_preparation_stage2/` — synthesize additional dialogue (TTS voices, identity QA, quality filtering, corpus mixing).
- **Advanced monitoring & evaluation** — alignment / audio-quality / dialogue / semantic monitors, sample savers, and enhanced eval metrics (`finetune/monitoring/`).
- **Staged training** — stage-1 pretraining and stage-2 speaker-conditioning recipes, single-node and FSDP/DDP (`example/korean_*.yaml`).

## Repository layout

```
finetune/
  backbone/         # pluggable LM backbone: base / factory / adapters / config
                    #   moshi_backbone.py (Moshi LM) + hf_lm_backbone.py (any HF causal LM)
  modules/          # zero-shot speaker conditioning: speaker_encoder / conditioner / audio_prompt
  monitoring/       # alignment / audio-quality / dialogue / semantic monitors + loggers
  data/             # dataset + interleaver (inner-monologue + dual audio streams)
  pretrained_loader.py, scheduler.py, train.py, eval.py, loss.py, args.py, ...
data_preparation/         # stage-1: real Korean dialogue -> Lhotse shards
data_preparation_stage2/  # stage-2: synthetic dialogue generation
example/                  # korean_*.yaml training/eval configs (FSDP / DDP / stages)
tools/                    # Korean tokenizer init / wrappers / conversion utilities
tests/                    # speaker-conditioning unit tests
docs/                     # architecture, speaker-conditioning, full-duplex, tokenizer, recipe notes
train.py, annotate.py
```

## Installation

```bash
# Python env (see pyproject.toml / uv.lock for pinned versions)
pip install -e .
# Base Moshi/Mimi weights are fetched from the kyutai-labs Hugging Face repos.
```

This recipe depends on the upstream **Kyutai Moshi** package and model weights; install/fetch them as documented by [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi).

## Data preparation

Point the configs at your own data — every dataset path is a `/path/to/...` placeholder. Korean dialogue can come from public corpora (e.g. AI-Hub broadcast/conversation sets, NIKL spoken corpus).

```bash
# Stage 1: real dialogue -> Lhotse shards
python -m data_preparation ...            # see data_preparation/README.md & ARCHITECTURE.md

# Stage 2 (optional): synthesize additional Korean dialogue
python -m data_preparation_stage2 ...     # see data_preparation_stage2/
```

## Training

```bash
# Stage-1 pretraining
python train.py example/korean_moshi_stage1_pretrain.yaml

# Stage-2 speaker conditioning
python train.py example/korean_moshi_stage2_speaker.yaml

# Multi-GPU FSDP / DDP
python train.py example/korean_v4_fsdp.yaml      # (or korean_*_ddp.yaml)
```

Edit the chosen `example/korean_*.yaml` to set model/data paths and choose the backbone (`moshi` or `hf_lm`).

## Evaluation & monitoring

```bash
python train.py example/korean_eval_speaker_cond.yaml   # speaker-conditioning eval
```

Training emits dialogue / semantic / alignment / audio-quality metrics and periodic sample dumps; see `finetune/monitoring/` and `docs/ENHANCED_EVAL_METRICS_SPEC.md`.

## Models & data (all public)

| Component | Source |
|-----------|--------|
| Base Moshi / Mimi codec | [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) (Hugging Face weights) |
| LM backbone | Moshi's native LM, or any public Hugging Face causal LM |
| Korean dialogue data | public corpora (e.g. AI-Hub, NIKL) — bring your own |

## Acknowledgements

- **[Kyutai Moshi & moshi-finetune](https://github.com/kyutai-labs/moshi)** — the base architecture and fine-tuning framework this work builds on.
- **[J-Moshi](https://arxiv.org/abs/2506.02979)** — Japanese Moshi adaptation that inspired this Korean effort.

## License & citation

Licensed under **Apache-2.0** (see [LICENSE](LICENSE)), matching upstream moshi-finetune. Please also credit the upstream projects above. The Moshi paper:

```bibtex
@techreport{kyutai2024moshi,
  title={Moshi: a speech-text foundation model for real-time dialogue},
  author={Défossez, Alexandre and Mazaré, Laurent and Orsini, Manu and others},
  institution={Kyutai},
  year={2024}
}
```
