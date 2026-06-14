#!/usr/bin/env python3
"""
=============================================================================
Step 6: Run Inference with Finetuned Model
=============================================================================

목적: 파인튜닝된 Moshi 모델로 추론을 실행합니다.
      - 체크포인트 구조 이해
      - LoRA 가중치 로딩 방법
      - Rust 서버 통합 가이드

입력: runs/tutorial_toy/checkpoints/ (Step 5에서 생성)
출력: 추론 결과 (오디오)

실행: python scripts/tutorials/06_inference.py
=============================================================================
"""

import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def explain_checkpoint_structure():
    """체크포인트 구조 설명"""
    print_banner("체크포인트 구조")

    print("""
[학습 출력 디렉토리 구조]
─────────────────────────────────────────────────────────────

  runs/tutorial_toy/
  ├── args.yaml                      # 학습 설정 복사본
  ├── train/                         # 학습 로그
  │   └── metrics.jsonl              # 스텝별 손실값
  ├── eval/                          # 평가 로그
  │   └── metrics.jsonl
  └── checkpoints/
      ├── checkpoint_000050/         # 50 스텝 체크포인트
      │   └── consolidated/
      │       ├── lora.safetensors   # LoRA 어댑터 가중치
      │       └── config.json        # 모델 설정
      └── checkpoint_000100/         # 100 스텝 체크포인트
          └── consolidated/
              ├── lora.safetensors
              └── config.json

[체크포인트 파일 설명]
─────────────────────────────────────────────────────────────

  lora.safetensors:
    - LoRA 어댑터 가중치만 저장 (~50-200MB)
    - 원본 모델과 합쳐서 사용

  consolidated.safetensors:
    - LoRA가 병합된 전체 모델 (~14GB)
    - 바로 추론에 사용 가능
    - save_adapters=false 시 생성

  config.json:
    - 모델 아키텍처 설정
    - LoRA 설정 (rank, scaling 등)
""")


def check_checkpoints(run_dir: Path):
    """체크포인트 확인"""
    print_banner("체크포인트 확인")

    ckpt_dir = run_dir / "checkpoints"

    if not ckpt_dir.exists():
        logger.warning(f"❌ 체크포인트 디렉토리가 없습니다: {ckpt_dir}")
        logger.info("   먼저 Step 5 (학습)을 실행해주세요.")
        return None

    # 체크포인트 목록
    checkpoints = sorted(ckpt_dir.glob("checkpoint_*"))

    if not checkpoints:
        logger.warning("❌ 체크포인트가 없습니다.")
        return None

    print(f"발견된 체크포인트: {len(checkpoints)}개\n")

    for ckpt in checkpoints:
        step = ckpt.name.split("_")[1]
        consolidated_dir = ckpt / "consolidated"

        lora_file = consolidated_dir / "lora.safetensors"
        full_file = consolidated_dir / "consolidated.safetensors"
        config_file = consolidated_dir / "config.json"

        print(f"  📁 {ckpt.name}/")

        if lora_file.exists():
            size_mb = lora_file.stat().st_size / 1024**2
            print(f"      ├─ lora.safetensors ({size_mb:.1f} MB)")

        if full_file.exists():
            size_gb = full_file.stat().st_size / 1024**3
            print(f"      ├─ consolidated.safetensors ({size_gb:.2f} GB)")

        if config_file.exists():
            print(f"      └─ config.json")

    # 최신 체크포인트 반환
    latest_ckpt = checkpoints[-1] / "consolidated"
    return latest_ckpt


def explain_lora_loading():
    """LoRA 로딩 방법 설명"""
    print_banner("LoRA 가중치 로딩 방법")

    print("""
[Python에서 LoRA 로딩]
─────────────────────────────────────────────────────────────

  from safetensors.torch import load_file
  from moshi.models import loaders

  # 1. 원본 Moshi 모델 로드 (LoRA 활성화)
  checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
      hf_repo="kyutai/moshiko-pytorch-bf16",
      moshi_weights="path/to/moshiko-pytorch-bf16",
      mimi_weights="path/to/tokenizer-*.safetensors",
      tokenizer="path/to/tokenizer_spm_32k_3.model",
  )

  # 모델 생성 (LoRA 활성화)
  lm_config = loaders._lm_kwargs
  lm_config["lora"] = True
  lm_config["lora_rank"] = 16  # 학습 시 사용한 rank
  lm_config["lora_scaling"] = 32  # 학습 시 사용한 alpha

  model = checkpoint_info.get_lm(device="cuda", **lm_config)

  # 2. LoRA 가중치 로드
  lora_weights = load_file("runs/.../lora.safetensors")
  model.load_state_dict(lora_weights, strict=False)

  # 3. 추론 모드
  model.eval()

[LoRA 병합 (Merged Model 생성)]
─────────────────────────────────────────────────────────────

  LoRA를 원본에 병합하여 단일 모델로 만들기:

  from moshi.modules.lora import merge_lora_weights

  # LoRA 가중치를 원본에 병합
  merged_state = merge_lora_weights(model)

  # 저장
  torch.save(merged_state, "merged_model.pt")

  장점:
  - 추론 시 추가 연산 없음
  - 배포 시 단일 파일
  - Rust 서버와 호환

[주의사항]
─────────────────────────────────────────────────────────────

  - lora_rank와 lora_scaling은 학습 시 사용한 값과 동일해야 함
  - config.json에서 설정 확인 가능
  - dtype도 맞춰야 함 (보통 bfloat16)
""")


def explain_rust_server_integration():
    """Rust 서버 통합 설명"""
    print_banner("Rust 서버 통합 가이드")

    print("""
[Moshi Rust 서버 개요]
─────────────────────────────────────────────────────────────

  Moshi는 Rust로 구현된 실시간 추론 서버를 제공합니다:

  moshi/                          # Rust 서버 프로젝트
  ├── rust/
  │   ├── moshi-core/            # 핵심 로직
  │   ├── moshi-backend/         # WebSocket 서버
  │   └── moshi-cli/             # CLI 도구
  └── client/                    # Web 클라이언트

[파인튜닝 모델을 Rust 서버에서 사용하기]
─────────────────────────────────────────────────────────────

  1. LoRA 병합된 모델 준비:

     학습 시 save_adapters=false로 설정하면
     consolidated.safetensors가 직접 생성됨

     또는 Python에서 수동 병합 후 저장

  2. Rust 서버 설정 수정:

     # moshi-backend 실행
     cd moshi/rust
     cargo run --release --bin moshi-backend -- \\
         --hf-repo kyutai/moshiko-pytorch-bf16 \\
         --moshi-weights /path/to/finetuned/consolidated.safetensors

  3. 클라이언트 접속:

     브라우저에서 http://localhost:8080 접속
     또는 WebSocket 클라이언트 사용

[한국어 K-Moshi 배포 흐름]
─────────────────────────────────────────────────────────────

  학습 완료
      ↓
  LoRA 병합 (consolidated.safetensors)
      ↓
  Rust 서버에 모델 경로 설정
      ↓
  서버 실행 & 클라이언트 접속
      ↓
  실시간 한국어 대화!

[MLX/GGML 변환 (선택)]
─────────────────────────────────────────────────────────────

  Apple Silicon에서 실행하려면 MLX 변환:

  # MLX 포맷으로 변환
  python -m moshi_mlx.convert \\
      --input consolidated.safetensors \\
      --output moshi-mlx/

  GGML 변환 (CPU 추론):

  # GGML 포맷으로 변환
  python scripts/convert_to_ggml.py \\
      --input consolidated.safetensors \\
      --output moshi.ggml
""")


def show_inference_example():
    """추론 예시 코드"""
    print_banner("추론 예시 코드")

    print("""
[기본 추론 (Python) - 로컬 경로 사용]
─────────────────────────────────────────────────────────────

  import json
  import torch
  from pathlib import Path
  from safetensors.torch import load_file
  from moshi.models import loaders

  # 로컬 경로 설정
  CKPT_DIR = Path("runs/tutorial_toy/checkpoints/checkpoint_000100/consolidated")
  MOSHI_WEIGHTS = Path("models/moshiko-pytorch-bf16/model.safetensors")
  MIMI_WEIGHTS = Path("models/tokenizer-e351c8d8-checkpoint125.safetensors")
  TOKENIZER = Path("models/tokenizer_spm_32k_3.model")

  # 설정 로드
  with open(CKPT_DIR / "config.json") as f:
      config = json.load(f)

  # CheckpointInfo 직접 생성 (로컬 경로 사용)
  checkpoint_info = loaders.CheckpointInfo(
      moshi_weights=MOSHI_WEIGHTS,
      mimi_weights=MIMI_WEIGHTS,
      tokenizer=TOKENIZER,
      lm_config=None,  # 기본 Moshi 7B 설정 사용
      raw_config=None,
  )

  # LoRA 설정 추가
  lm_config = dict(loaders._lm_kwargs)
  lm_config.update({
      "lora": config.get("lora", True),
      "lora_rank": config.get("lora_rank", 8),
      "lora_scaling": config.get("lora_scaling", 2.0),
  })

  # 모델 로드
  model = loaders.get_moshi_lm(
      filename=str(MOSHI_WEIGHTS),
      lm_kwargs=lm_config,
      device="cuda",
      dtype=torch.bfloat16,
  )

  # LoRA 가중치 로드
  lora_path = CKPT_DIR / "lora.safetensors"
  if lora_path.exists():
      lora_weights = load_file(str(lora_path), device="cuda")
      model.load_state_dict(lora_weights, strict=False)

  model.eval()

  # Mimi 코덱 로드
  mimi = checkpoint_info.get_mimi(device="cuda")
  mimi.eval()

  # 토크나이저 로드
  spm = checkpoint_info.get_text_tokenizer()

  print("✅ 모델 로딩 완료!")
  print(f"   LoRA rank: {config.get('lora_rank')}")
  print(f"   LoRA scaling: {config.get('lora_scaling')}")

[스트리밍 추론 (실시간)]
─────────────────────────────────────────────────────────────

  실시간 대화를 위해서는 Rust 서버 사용을 권장합니다.
  Python에서의 스트리밍 구현은 moshi 라이브러리의
  streaming 모듈을 참조하세요.

[mpirun + DDP 설정으로 학습한 모델 사용]
─────────────────────────────────────────────────────────────

  이 튜토리얼은 mpirun + DDP 설정을 기본으로 합니다:

  - 설정: distributed_backend: ddp
  - 런처: mpirun (05_train_single_gpu.sh)
  - 학습: bash scripts/train_mpi.sh --config ...

  생성된 체크포인트는 torchrun/FSDP로 학습한 것과
  동일하게 사용할 수 있습니다.
""")


def main():
    """메인 실행 함수"""
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent
    os.chdir(project_root)

    logger.info(f"📍 작업 디렉토리: {os.getcwd()}")

    # 체크포인트 구조 설명
    explain_checkpoint_structure()

    # 체크포인트 확인
    run_dir = Path("runs/tutorial_toy")
    latest_ckpt = check_checkpoints(run_dir)

    # LoRA 로딩 방법 설명
    explain_lora_loading()

    # Rust 서버 통합 설명
    explain_rust_server_integration()

    # 추론 예시
    show_inference_example()

    print_banner("튜토리얼 완료!")

    print("""
[다음 단계]
─────────────────────────────────────────────────────────────

  🎉 축하합니다! Moshi 파인튜닝 튜토리얼을 완료했습니다.

  한국어 K-Moshi 구축을 위한 다음 단계:

  1. 한국어 데이터 수집 및 전처리
     - KsponSpeech (969시간) 권장
     - annotate.py --lang ko 사용

  2. 대규모 학습 실행
     - A100 80GB에서 다중 GPU 학습
     - configs/tutorials/korean_finetune_template.yaml 참조

  3. 모델 평가 및 튜닝
     - 한국어 음성 인식 품질 평가
     - LoRA rank, 학습률 조정

  4. Rust 서버 배포
     - 파인튜닝된 모델로 실시간 서비스

  💡 도움이 필요하면 CLAUDE.md를 참조하세요!
""")

    if latest_ckpt:
        logger.info("✅ 튜토리얼 완료")
        logger.info(f"   최신 체크포인트: {latest_ckpt}")
    else:
        logger.info("ℹ️  튜토리얼 설명 완료")
        logger.info("   학습을 실행하면 체크포인트가 생성됩니다.")


if __name__ == "__main__":
    main()
