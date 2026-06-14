#!/usr/bin/env python3
"""
=============================================================================
Step 4: Download Pretrained Model
=============================================================================

목적: Moshi 파인튜닝에 필요한 사전학습 모델을 다운로드합니다.
      - Moshi 모델 (7B 파라미터)
      - Mimi 오디오 코덱
      - SentencePiece 토크나이저

입력: 없음 (HuggingFace에서 다운로드)
출력: models/ 디렉토리에 모델 파일 저장

실행: python scripts/tutorials/04_download_model.py
=============================================================================
"""

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


def explain_model_architecture():
    """Moshi 모델 아키텍처 설명"""
    print_banner("Moshi 모델 아키텍처")

    print("""
[Moshi 전체 구조]
─────────────────────────────────────────────────────────────

                    ┌─────────────────────────────────────┐
                    │         Moshi (7B LLM)              │
                    │                                     │
  Text Stream ─────►│  ┌─────────────────────────────┐   │
  (Inner Monologue) │  │   Transformer Decoder       │   │
                    │  │   - 32 layers               │   │
                    │  │   - 4096 hidden dim         │   │
                    │  │   - 32 attention heads      │   │
  Audio Streams ───►│  │   - RoPE positional enc    │   │
  (8 codebooks)     │  └─────────────────────────────┘   │
                    │                                     │
                    │  Input: 9 streams (1 text + 8 audio)│
                    │  Output: Next token prediction      │
                    └─────────────────────────────────────┘

[Mimi 오디오 코덱]
─────────────────────────────────────────────────────────────

  Audio (24kHz) ──► Encoder ──► Quantizer ──► 8 Codebooks
                                    │
                                    ▼
                              ┌──────────┐
                              │ Codebook │
                              │ Index    │
                              │ (0-2047) │
                              └──────────┘

  특성:
    - 프레임 레이트: 12.5 Hz (80ms per frame)
    - 코드북 수: 8 (RVQ)
    - 코드북 크기: 2048 entries each
    - 비트레이트: ~1.1 kbps

[토크나이저]
─────────────────────────────────────────────────────────────

  SentencePiece (Unigram)
    - Vocab size: 32,000
    - 영어 중심 + 다국어 지원
    - 한국어: 자모 분리로 처리

[모델 파일 목록]
─────────────────────────────────────────────────────────────

  models/
  ├── moshiko-pytorch-bf16/           # Moshi 모델
  │   ├── model.safetensors           # ~14GB (bf16)
  │   └── config.json                 # 모델 설정
  │
  ├── tokenizer-e351c8d8-*.safetensors  # Mimi 코덱
  │                                      # ~100MB
  │
  └── tokenizer_spm_32k_3.model         # 토크나이저
                                         # ~1MB
""")


def download_models(models_dir: Path):
    """HuggingFace에서 모델 다운로드"""
    print_banner("모델 다운로드")

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        logger.error("huggingface_hub이 설치되지 않았습니다.")
        logger.info("설치: pip install huggingface_hub")
        sys.exit(1)

    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Moshi 모델 다운로드
    moshi_dir = models_dir / "moshiko-pytorch-bf16"
    if moshi_dir.exists():
        logger.info(f"✅ Moshi 모델 이미 존재: {moshi_dir}")
    else:
        logger.info("📥 Moshi 모델 다운로드 중... (약 14GB)")
        logger.info("   이 작업은 시간이 걸릴 수 있습니다.")

        try:
            snapshot_download(
                "kyutai/moshiko-pytorch-bf16",
                local_dir=str(moshi_dir),
                local_dir_use_symlinks=False,
            )
            logger.info(f"✅ Moshi 모델 다운로드 완료: {moshi_dir}")
        except Exception as e:
            logger.error(f"❌ Moshi 다운로드 실패: {e}")
            logger.info("💡 수동 다운로드: https://huggingface.co/kyutai/moshiko-pytorch-bf16")
            return False

    # 2. Mimi 코덱 다운로드
    mimi_file = models_dir / "tokenizer-e351c8d8-checkpoint125.safetensors"
    if mimi_file.exists():
        logger.info(f"✅ Mimi 코덱 이미 존재: {mimi_file}")
    else:
        logger.info("📥 Mimi 코덱 다운로드 중... (약 100MB)")

        try:
            hf_hub_download(
                "kyutai/moshiko-pytorch-bf16",
                filename="tokenizer-e351c8d8-checkpoint125.safetensors",
                local_dir=str(models_dir),
            )
            logger.info(f"✅ Mimi 코덱 다운로드 완료: {mimi_file}")
        except Exception as e:
            logger.error(f"❌ Mimi 다운로드 실패: {e}")
            return False

    # 3. 토크나이저 다운로드
    tokenizer_file = models_dir / "tokenizer_spm_32k_3.model"
    if tokenizer_file.exists():
        logger.info(f"✅ 토크나이저 이미 존재: {tokenizer_file}")
    else:
        logger.info("📥 토크나이저 다운로드 중... (약 1MB)")

        try:
            hf_hub_download(
                "kyutai/moshiko-pytorch-bf16",
                filename="tokenizer_spm_32k_3.model",
                local_dir=str(models_dir),
            )
            logger.info(f"✅ 토크나이저 다운로드 완료: {tokenizer_file}")
        except Exception as e:
            logger.error(f"❌ 토크나이저 다운로드 실패: {e}")
            return False

    return True


def verify_models(models_dir: Path):
    """다운로드된 모델 검증"""
    print_banner("모델 검증")

    required_files = [
        ("moshiko-pytorch-bf16", "Moshi 모델 디렉토리"),
        ("tokenizer-e351c8d8-checkpoint125.safetensors", "Mimi 코덱"),
        ("tokenizer_spm_32k_3.model", "SentencePiece 토크나이저"),
    ]

    all_present = True
    for filename, description in required_files:
        path = models_dir / filename
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {description}: {path}")
        if not exists:
            all_present = False

    if all_present:
        # 파일 크기 출력
        print("\n[파일 크기]")
        print("-" * 50)

        moshi_dir = models_dir / "moshiko-pytorch-bf16"
        if moshi_dir.is_dir():
            total_size = sum(f.stat().st_size for f in moshi_dir.rglob("*") if f.is_file())
            print(f"  Moshi 모델: {total_size / 1024**3:.2f} GB")

        mimi_file = models_dir / "tokenizer-e351c8d8-checkpoint125.safetensors"
        if mimi_file.exists():
            print(f"  Mimi 코덱: {mimi_file.stat().st_size / 1024**2:.2f} MB")

        tok_file = models_dir / "tokenizer_spm_32k_3.model"
        if tok_file.exists():
            print(f"  토크나이저: {tok_file.stat().st_size / 1024**2:.2f} MB")

    return all_present


def explain_gpu_requirements():
    """GPU 요구사항 설명"""
    print_banner("GPU 요구사항")

    print("""
[메모리 요구사항]
─────────────────────────────────────────────────────────────

  Moshi 7B (bf16):
    - 모델 가중치: ~14 GB
    - LoRA 어댑터: ~50-200 MB (rank에 따라)
    - Optimizer 상태: ~1-2 GB
    - Activation: 배치/시퀀스 길이에 따라

  최소 GPU 메모리:
    - A100 80GB: 권장 (batch_size=2, duration=100)
    - A100 40GB: 가능 (batch_size=1, duration=60)
    - V100 32GB: 제한적 (batch_size=1, duration=30)

[메모리 최적화 옵션]
─────────────────────────────────────────────────────────────

  1. Duration 줄이기:
     - 100초 → 30초로 줄이면 메모리 ~60% 감소
     - 단, 긴 대화 패턴 학습 어려움

  2. Batch size 줄이기:
     - Gradient accumulation으로 보상 가능
     - 학습 안정성 주의

  3. LoRA rank 줄이기:
     - rank 64 → 8로 줄이면 어댑터 메모리 ~87% 감소
     - 표현력 감소 trade-off

  4. Gradient checkpointing:
     - 메모리 절약 but 학습 속도 ~20% 감소

[튜토리얼 권장 설정]
─────────────────────────────────────────────────────────────

  빠른 테스트 (V100 32GB):
    - duration: 30
    - batch_size: 1
    - lora_rank: 8
    - max_steps: 100

  실제 학습 (A100 80GB):
    - duration: 100
    - batch_size: 2
    - lora_rank: 32
    - max_steps: 50000+
""")


def main():
    """메인 실행 함수"""
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent
    os.chdir(project_root)

    logger.info(f"📍 작업 디렉토리: {os.getcwd()}")

    models_dir = Path("models")

    # 아키텍처 설명
    explain_model_architecture()

    # 모델 다운로드
    success = download_models(models_dir)

    if success:
        # 모델 검증
        verify_models(models_dir)

    # GPU 요구사항 설명
    explain_gpu_requirements()

    print_banner("Step 4 완료!")
    if success:
        logger.info("✅ 모델 다운로드 및 검증 완료")
        logger.info("➡️  다음 단계: python scripts/tutorials/05_train_single_gpu.py")
    else:
        logger.warning("⚠️  일부 모델 다운로드 실패")
        logger.info("   위의 오류 메시지를 확인하고 수동으로 다운로드해주세요.")


if __name__ == "__main__":
    main()
