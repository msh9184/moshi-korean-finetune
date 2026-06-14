# Stage-Based Pretrained Model Loading 설계서

## 1. 개요 및 요구사항

### 1.1 배경
K-Moshi 학습은 여러 stage로 진행됩니다:
- **Stage 1**: Speaker conditioning 없이 한국어 데이터셋으로 기본 학습
- **Stage 2+**: Speaker conditioning (encoder + audio prompt) 추가하여 학습

Stage 1에서 학습된 모델의 weights를 Stage 2 학습의 초기값으로 사용하되,
새로 추가된 파라미터(speaker_conditioner 등)는 랜덤 초기화하고 step=0부터 학습을 시작해야 합니다.

### 1.2 요구사항
1. **safetensors 파일 로딩**: 이전 stage에서 저장된 `.safetensors` 파일 로딩
2. **부분 파라미터 매칭**: 존재하는 key만 로딩, 새 파라미터는 초기화 유지
3. **Step 0 시작**: training state 무시, step=0부터 새로 학습
4. **FSDP 호환성**: FullyShardedDataParallel과 호환되는 로딩 방식
5. **Checkpoint Resume과의 구분**: 기존 resume 기능과 명확히 분리

### 1.3 현재 Checkpoint Resume과의 차이점

| 특성 | Checkpoint Resume | Stage Pretrained Loading |
|------|-------------------|--------------------------|
| **목적** | 중단된 학습 재개 | 새 stage 학습 시작 |
| **Step** | 저장된 step에서 계속 | Step 0부터 시작 |
| **Optimizer** | 복원 | 새로 초기화 |
| **Scheduler** | 복원 | 새로 초기화 |
| **RNG State** | 복원 | 새 seed 사용 |
| **새 파라미터** | 에러 (strict loading) | 랜덤 초기화 유지 |
| **누락 파라미터** | 에러 (strict loading) | 무시 (warning) |

## 2. 아키텍처 분석

### 2.1 현재 Checkpoint 구조
```
runs/{run_name}/checkpoints/
├── config.json                                           # Model config
├── checkpoint.eval_loss-2.434.step-000880.safetensors   # Model weights
├── checkpoint.eval_loss-2.434.step-000880.best.safetensors  # Symlink → best
├── checkpoint.eval_loss-2.434.step-000880.last.safetensors  # Symlink → latest
├── training_state.step-000880.last.pt                   # Training state
└── training_state.step-000880.best.pt                   # Training state
```

### 2.2 safetensors 파일의 Weight Key 구조
```python
# LMModel (Moshi backbone) weights
"transformer.layers.0.gating.linear_in.weight"
"transformer.layers.0.self_attn.out_proj.weight"
"transformer.layers.0.self_attn.in_proj_weight"
...

# Depformer weights
"depformer.layers.0.self_attn.in_proj_weight"
...

# Embedding weights
"text_emb.weight"
"audio_embs.0.weight"
...

# Output linear weights
"linears.0.weight"
"text_linear.weight"
...
```

### 2.3 새로 추가되는 파라미터 (Stage 2)
```python
# Speaker Conditioner (새로 추가됨)
"speaker_conditioner.projection.weight"      # [4096, 256]
"speaker_conditioner.projection.bias"        # [4096]
"speaker_conditioner.layernorm.weight"       # [4096]
"speaker_conditioner.layernorm.bias"         # [4096]
"speaker_conditioner.scale"                  # scalar

# Speaker Encoder는 freeze=True이므로 학습 대상 아님
```

## 3. 설계

### 3.1 새 Config 옵션 추가

```yaml
# finetune/args.py에 추가
@dataclass
class PretrainedModelArgs(Serializable):
    """Stage-based pretrained model loading configuration."""

    # Enable pretrained model loading (from previous stage)
    enabled: bool = False

    # Path to safetensors file from previous stage
    # Supports:
    # - Absolute path: "/path/to/checkpoint.safetensors"
    # - Relative to run_dir: "checkpoints/checkpoint.best.safetensors"
    # - "best" or "last" keywords: Auto-find in specified checkpoint_dir
    path: str | None = None

    # Directory containing checkpoints (for "best"/"last" keywords)
    checkpoint_dir: str | None = None

    # Strict loading mode:
    # - True: Error on missing keys (except expected new modules)
    # - False: Warning only, continue with partial loading
    strict: bool = False

    # List of module prefixes expected to be newly initialized
    # These won't trigger warnings when missing in pretrained weights
    expected_new_modules: list[str] = field(default_factory=lambda: [
        "speaker_conditioner",
        "dimension_adapter",
    ])

    # Log detailed loading information
    verbose: bool = True
```

### 3.2 YAML 설정 예시

```yaml
# example/korean_moshi_stage2_speaker.yaml

# Stage 1에서 학습된 모델 로딩
pretrained:
  enabled: true
  path: "best"  # runs/korean_moshi_stage1_pretrain/checkpoints에서 best 찾기
  checkpoint_dir: "./runs/korean_moshi_stage1_pretrain/checkpoints"
  strict: false
  expected_new_modules:
    - "speaker_conditioner"
  verbose: true

# Speaker conditioning 활성화 (Stage 2)
speaker:
  enabled: true
  method: "both"
  encoder:
    encoder_type: "w2v_bert2"
    ...
```

### 3.3 로딩 함수 구현

```python
# finetune/pretrained_loader.py (새 파일)

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Set

import safetensors.torch
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from .args import PretrainedModelArgs

logger = logging.getLogger("pretrained_loader")


def load_pretrained_weights(
    model: torch.nn.Module,
    args: PretrainedModelArgs,
    run_dir: Optional[Path] = None,
) -> Tuple[int, int, Set[str]]:
    """
    Load pretrained weights from a previous training stage.

    This function performs PARTIAL weight loading:
    - Loads weights that exist in both model and checkpoint
    - Skips weights that only exist in checkpoint (removed modules)
    - Leaves newly added parameters with their initial values

    Args:
        model: Target model to load weights into (may be FSDP-wrapped)
        args: Pretrained model loading configuration
        run_dir: Current run directory for relative path resolution

    Returns:
        Tuple of (loaded_count, skipped_count, new_params)
        - loaded_count: Number of parameters successfully loaded
        - skipped_count: Number of checkpoint params not in model
        - new_params: Set of model param names not in checkpoint
    """
    if not args.enabled:
        return 0, 0, set()

    # Resolve checkpoint path
    ckpt_path = _resolve_checkpoint_path(args, run_dir)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"Could not find pretrained checkpoint. "
            f"path={args.path}, checkpoint_dir={args.checkpoint_dir}"
        )

    logger.info(f"[PRETRAINED] Loading weights from: {ckpt_path}")

    # Load checkpoint weights
    ckpt_state_dict = safetensors.torch.load_file(ckpt_path, device="cpu")

    # Get model state dict (handle FSDP)
    model_state_dict = _get_model_state_dict(model)

    # Compute key differences
    ckpt_keys = set(ckpt_state_dict.keys())
    model_keys = set(model_state_dict.keys())

    # Keys in both
    common_keys = ckpt_keys & model_keys

    # Keys only in checkpoint (will be skipped)
    ckpt_only_keys = ckpt_keys - model_keys

    # Keys only in model (new parameters)
    model_only_keys = model_keys - ckpt_keys

    # Filter out expected new modules
    expected_new_keys = set()
    unexpected_new_keys = set()
    for key in model_only_keys:
        is_expected = any(
            key.startswith(prefix)
            for prefix in args.expected_new_modules
        )
        if is_expected:
            expected_new_keys.add(key)
        else:
            unexpected_new_keys.add(key)

    # Log statistics
    if args.verbose:
        logger.info(f"[PRETRAINED] Checkpoint keys: {len(ckpt_keys)}")
        logger.info(f"[PRETRAINED] Model keys: {len(model_keys)}")
        logger.info(f"[PRETRAINED] Common keys (to load): {len(common_keys)}")
        logger.info(f"[PRETRAINED] Checkpoint-only keys (skipped): {len(ckpt_only_keys)}")
        logger.info(f"[PRETRAINED] Expected new params: {len(expected_new_keys)}")

        if ckpt_only_keys:
            logger.warning(
                f"[PRETRAINED] Skipping {len(ckpt_only_keys)} checkpoint keys not in model: "
                f"{list(ckpt_only_keys)[:5]}..."
            )

        if expected_new_keys:
            logger.info(
                f"[PRETRAINED] New modules (randomly initialized): "
                f"{list(expected_new_keys)[:5]}..."
            )

    # Handle unexpected new keys
    if unexpected_new_keys:
        msg = (
            f"[PRETRAINED] Found {len(unexpected_new_keys)} unexpected new parameters "
            f"not in checkpoint: {list(unexpected_new_keys)[:5]}..."
        )
        if args.strict:
            raise RuntimeError(msg)
        else:
            logger.warning(msg)

    # Create partial state dict for loading
    partial_state_dict = {
        k: ckpt_state_dict[k]
        for k in common_keys
    }

    # Load weights into model
    _load_state_dict_partial(model, partial_state_dict)

    logger.info(
        f"[PRETRAINED] Successfully loaded {len(common_keys)} parameters "
        f"from {ckpt_path.name}"
    )

    return len(common_keys), len(ckpt_only_keys), model_only_keys


def _resolve_checkpoint_path(
    args: PretrainedModelArgs,
    run_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Resolve checkpoint path from config."""

    if args.path is None:
        return None

    # Handle "best" or "last" keywords
    if args.path.lower() in ("best", "last"):
        if args.checkpoint_dir is None:
            logger.error(
                f"checkpoint_dir must be specified when path='{args.path}'"
            )
            return None

        ckpt_dir = Path(args.checkpoint_dir)
        if not ckpt_dir.is_absolute() and run_dir:
            ckpt_dir = run_dir / args.checkpoint_dir

        return _find_checkpoint_by_tag(ckpt_dir, args.path.lower())

    # Handle absolute path
    path = Path(args.path)
    if path.is_absolute():
        return path if path.exists() else None

    # Handle relative path (to checkpoint_dir or run_dir)
    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir)
        if not ckpt_dir.is_absolute() and run_dir:
            ckpt_dir = run_dir / args.checkpoint_dir
        candidate = ckpt_dir / args.path
        if candidate.exists():
            return candidate

    if run_dir:
        candidate = run_dir / args.path
        if candidate.exists():
            return candidate

    return None


def _find_checkpoint_by_tag(ckpt_dir: Path, tag: str) -> Optional[Path]:
    """Find checkpoint by 'best' or 'last' symlink."""

    if not ckpt_dir.exists():
        return None

    # Look for symlink
    for f in ckpt_dir.iterdir():
        if f.name.endswith(f".{tag}.safetensors"):
            if f.is_symlink():
                target = f.resolve()
                if target.exists():
                    return target
            elif f.exists():
                return f

    # Fallback: find by parsing filenames
    import re
    pattern = re.compile(
        r"^.+\.[\w_]+-[\d.]+\.step-(\d+)\.safetensors$"
    )

    checkpoints = []
    for f in ckpt_dir.iterdir():
        if f.suffix == ".safetensors" and not f.is_symlink():
            if not f.name.endswith((".best.safetensors", ".last.safetensors")):
                match = pattern.match(f.name)
                if match:
                    step = int(match.group(1))
                    checkpoints.append((step, f))

    if not checkpoints:
        return None

    # Sort by step
    checkpoints.sort(key=lambda x: x[0], reverse=True)

    # "last" = highest step, "best" would need metric parsing (fallback to last)
    return checkpoints[0][1]


def _get_model_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Get model state dict, handling FSDP wrapping."""

    if isinstance(model, FSDP):
        # For FSDP, we need to use summon_full_params for state dict access
        # But for key comparison, we can use _fsdp_wrapped_module
        if hasattr(model, '_fsdp_wrapped_module'):
            inner = model._fsdp_wrapped_module
        elif hasattr(model, 'module'):
            inner = model.module
        else:
            inner = model
        return dict(inner.state_dict())

    return dict(model.state_dict())


def _load_state_dict_partial(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> None:
    """Load state dict with partial matching (non-strict)."""

    if isinstance(model, FSDP):
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        full_state_dict_config = FullStateDictConfig(
            offload_to_cpu=True,
            rank0_only=False
        )
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            full_state_dict_config
        ):
            model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
```

### 3.4 train.py 통합

```python
# train.py에 추가 (model 로딩 후)

# 4.3 Load pretrained weights from previous stage (if configured)
if hasattr(args, 'pretrained') and args.pretrained.enabled:
    from finetune.pretrained_loader import load_pretrained_weights

    main_logger_info("=" * 60)
    main_logger_info("[STAGE PRETRAINED LOADING]")

    loaded, skipped, new_params = load_pretrained_weights(
        model=model,
        args=args.pretrained,
        run_dir=Path(args.run_dir).parent if args.run_dir else None,
    )

    main_logger_info(f"  Loaded parameters: {loaded}")
    main_logger_info(f"  Skipped (not in model): {skipped}")
    main_logger_info(f"  New parameters (initialized): {len(new_params)}")
    main_logger_info("=" * 60)

# Checkpoint Manager should NOT resume when pretrained loading is used
if hasattr(args, 'pretrained') and args.pretrained.enabled:
    # Force disable resume to start from step 0
    args.checkpoint.resume_if_exist = False
    args.checkpoint.resume_from = None
    main_logger_info("[PRETRAINED] Resume disabled - starting from step 0")
```

## 4. 검증 체크리스트

### 4.1 기능 검증
- [ ] safetensors 파일 로딩 성공
- [ ] 존재하는 key만 로딩 (partial matching)
- [ ] 새 파라미터(speaker_conditioner)는 초기화 유지
- [ ] Step 0부터 학습 시작
- [ ] Optimizer/Scheduler 새로 초기화
- [ ] FSDP 호환성 확인

### 4.2 에러 케이스 검증
- [ ] 잘못된 경로 → FileNotFoundError
- [ ] strict=True + unexpected new keys → RuntimeError
- [ ] 빈 checkpoint_dir → 적절한 에러 메시지

### 4.3 로그 검증
- [ ] 로딩된 파라미터 수 출력
- [ ] 새 파라미터 목록 출력
- [ ] 스킵된 파라미터 경고

## 5. 사용 예시

### 5.1 Stage 1 학습 (기본)
```bash
torchrun --nproc-per-node 8 -m train example/korean_moshi_stage1_pretrain.yaml
# Output: runs/korean_moshi_stage1_pretrain/checkpoints/checkpoint.*.safetensors
```

### 5.2 Stage 2 학습 (Speaker Conditioning 추가)
```yaml
# example/korean_moshi_stage2_speaker.yaml
pretrained:
  enabled: true
  path: "best"
  checkpoint_dir: "./runs/korean_moshi_stage1_pretrain/checkpoints"

speaker:
  enabled: true
  method: "both"
  ...
```

```bash
torchrun --nproc-per-node 8 -m train example/korean_moshi_stage2_speaker.yaml
# - Stage 1 weights 로딩
# - speaker_conditioner는 랜덤 초기화 유지
# - Step 0부터 학습 시작
```

## 6. 향후 확장 고려사항

### 6.1 Multiple Checkpoint Merging
여러 checkpoint를 병합하는 기능 (예: backbone + speaker encoder weights)

### 6.2 Selective Layer Loading
특정 layer만 로딩/동결하는 기능

### 6.3 Weight Interpolation
두 checkpoint 사이의 interpolation (model soup)

---
*Author: K-Moshi Development Team*
*Date: 2026-01-23*
*Version: 1.0*
