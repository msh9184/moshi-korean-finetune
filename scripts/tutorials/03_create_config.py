#!/usr/bin/env python3
"""
=============================================================================
Step 3: Create Training Configuration
=============================================================================

목적: Moshi finetuning을 위한 YAML 설정 파일을 생성합니다.
      - TrainArgs 구조 이해
      - LoRA 설정 파라미터 설명
      - GPU 환경별 최적화 설정

입력: 없음 (설정값 정의)
출력: configs/tutorials/toy_finetune.yaml

실행: python scripts/tutorials/03_create_config.py
=============================================================================
"""

import logging
import os
from pathlib import Path

import yaml

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


def explain_train_args():
    """TrainArgs 파라미터 설명"""
    print_banner("TrainArgs 파라미터 설명")

    print("""
[데이터 설정 - data (DataArgs)]
─────────────────────────────────────────────────────────────
  data.train_data  : 학습 데이터 JSONL 파일 경로
  data.eval_data   : 평가 데이터 JSONL 파일 경로 (선택)
  data.shuffle     : 데이터 셔플 여부

[모델 경로 - moshi_paths (ModelPaths)]
─────────────────────────────────────────────────────────────
  moshi_paths.hf_repo_id   : HuggingFace 레포 ID (로컬 시 None)
  moshi_paths.moshi_path   : Moshi 모델 체크포인트 경로
  moshi_paths.mimi_path    : Mimi 오디오 코덱 체크포인트 경로
  moshi_paths.tokenizer_path : 텍스트 토크나이저 경로

[기본 설정]
─────────────────────────────────────────────────────────────
  run_dir          : 학습 출력 디렉토리 (체크포인트, 로그 저장)
  duration_sec     : 청크 길이 (초). Mimi 프레임 레이트(12.5Hz) 고려
                     - 10-30초 권장 (메모리에 맞게 조절)
  batch_size       : GPU당 배치 크기 (A100: 2, V100: 1)
  max_steps        : 최대 학습 스텝

[옵티마이저 - optim (OptimArgs)]
─────────────────────────────────────────────────────────────
  optim.lr          : 학습률 (LoRA: 1e-4 ~ 2e-4 권장)
  optim.weight_decay: 가중치 감쇠 (0.1 기본)
  optim.pct_start   : OneCycleLR 웜업 비율 (0.05 = 5%)

[LoRA 설정 - lora (LoraArgs)]
─────────────────────────────────────────────────────────────
  lora.enable      : LoRA 활성화 여부
  lora.rank        : LoRA 행렬 랭크 (8-64, 클수록 표현력↑ 메모리↑)
  lora.scaling     : LoRA 스케일링 (보통 2.0)
  lora.ft_embed    : 임베딩 레이어도 파인튜닝 여부

[분산 학습]
─────────────────────────────────────────────────────────────
  distributed_backend: 분산 학습 백엔드
                       - "fsdp": torchrun 사용 시 (기본)
                       - "ddp": mpirun 사용 시 권장

[손실 함수 가중치]
─────────────────────────────────────────────────────────────
  first_codebook_weight_multiplier: 첫 코드북 가중치 (1.0 기본)
  text_padding_weight: 텍스트 패딩 가중치 (0.5 기본)
""")


def explain_lora():
    """LoRA 개념 설명"""
    print_banner("LoRA (Low-Rank Adaptation) 이해")

    print("""
[LoRA란?]
─────────────────────────────────────────────────────────────
  대규모 모델을 효율적으로 파인튜닝하는 기법입니다.
  원본 가중치는 동결하고, 작은 크기의 어댑터만 학습합니다.

[수학적 원리]
─────────────────────────────────────────────────────────────
  원본: W ∈ R^(d×k)
  LoRA: W' = W + BA
        - B ∈ R^(d×r), A ∈ R^(r×k)
        - r << min(d, k) (저차원)

  예시 (r=16):
    - 원본 파라미터: d×k = 4096×4096 = 16.7M
    - LoRA 파라미터: d×r + r×k = 4096×16 + 16×4096 = 131K
    - 압축률: 99.2% 파라미터 감소!

[Moshi에서의 LoRA 적용]
─────────────────────────────────────────────────────────────
  적용 대상:
    - self_attn: Q, K, V projection
    - mlp: up, down, gate projection

  Moshi 7B 모델:
    - 전체 파라미터: ~7B
    - LoRA(r=16) 파라미터: ~50M (0.7%)
    - LoRA(r=64) 파라미터: ~200M (2.8%)

[권장 설정]
─────────────────────────────────────────────────────────────
  빠른 실험:
    lora_rank: 8, lora_alpha: 16

  균형 잡힌 설정:
    lora_rank: 16, lora_alpha: 32

  고품질 (충분한 데이터 시):
    lora_rank: 64, lora_alpha: 128
""")


def create_toy_config(config_dir: Path) -> Path:
    """튜토리얼용 설정 파일 생성 (TrainArgs 구조에 맞춤)"""
    print_banner("튜토리얼 설정 파일 생성")

    # TrainArgs 구조에 맞는 설정
    config = {
        # 데이터 설정 (DataArgs)
        # 참고: HuggingFace DailyTalkContiguous 데이터셋은 dailytalk.jsonl 파일명 사용
        "data": {
            "train_data": "data/daily-talk-contiguous/dailytalk.jsonl",
            "eval_data": "",
            "shuffle": True,
        },

        # 출력 디렉토리
        "run_dir": "runs/tutorial_toy",

        # 모델 경로 (ModelPaths)
        "moshi_paths": {
            "hf_repo_id": None,  # 로컬 경로 사용 시 None
            "moshi_path": "models/moshiko-pytorch-bf16/model.safetensors",
            "mimi_path": "models/tokenizer-e351c8d8-checkpoint125.safetensors",
            "tokenizer_path": "models/tokenizer_spm_32k_3.model",
            "config_path": None,
        },

        # 학습 설정
        "duration_sec": 10.0,  # 튜토리얼용으로 짧게 설정 (10초)
        "batch_size": 1,
        "max_steps": 100,  # 튜토리얼용으로 짧게
        "log_freq": 10,
        "ckpt_freq": 50,
        "num_microbatches": 1,
        "max_norm": 1.0,

        # 옵티마이저 설정 (OptimArgs)
        "optim": {
            "lr": 0.0001,
            "weight_decay": 0.1,
            "pct_start": 0.05,
        },

        # LoRA 설정 (LoraArgs)
        "lora": {
            "enable": True,
            "rank": 8,
            "scaling": 2.0,
            "ft_embed": False,
        },

        # 손실 함수 가중치
        "first_codebook_weight_multiplier": 1.0,
        "text_padding_weight": 0.5,

        # 체크포인팅
        "do_ckpt": True,
        "save_adapters": True,
        "num_ckpt_keep": 3,

        # 평가
        "do_eval": False,
        "eval_freq": 0,

        # 효율성
        "gradient_checkpointing": True,
        "param_dtype": "bfloat16",

        # 분산 학습 - mpirun 사용 시 ddp 권장
        "distributed_backend": "ddp",  # fsdp (torchrun) 또는 ddp (mpirun)

        # WandB (비활성화)
        "wandb": {
            "project": None,
            "offline": False,
        },

        # 기타
        "seed": 42,
        "full_finetuning": False,
        "overwrite_run_dir": True,
    }

    config_path = config_dir / "toy_finetune.yaml"

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    logger.info(f"📄 설정 파일 생성: {config_path}")

    # 설정 내용 출력
    print("\n[생성된 설정 파일 내용]")
    print("-" * 50)
    with open(config_path) as f:
        print(f.read())

    return config_path


def create_korean_config_template(config_dir: Path) -> Path:
    """한국어 파인튜닝용 설정 템플릿 생성 (TrainArgs 구조에 맞춤)"""
    print_banner("한국어 파인튜닝 설정 템플릿")

    # TrainArgs 구조에 맞는 설정
    config = {
        # 데이터 설정 (DataArgs)
        "data": {
            "train_data": "/path/to",
            "eval_data": "/path/to",
            "shuffle": True,
        },

        # 출력 디렉토리
        "run_dir": "/path/to",

        # 모델 경로 (ModelPaths) - GPU 서버 기준
        "moshi_paths": {
            "hf_repo_id": None,  # 로컬 경로 사용
            "moshi_path": "/path/to",
            "mimi_path": "/path/to",
            "tokenizer_path": "/path/to",
            "config_path": None,
        },

        # 학습 설정 (A100 80GB 기준)
        "duration_sec": 30.0,  # 30초 청크 (메모리 고려)
        "batch_size": 2,    # A100 80GB에서 2 가능
        "max_steps": 50000,  # 데이터 양에 따라 조절
        "log_freq": 50,
        "ckpt_freq": 1000,
        "num_microbatches": 1,
        "max_norm": 1.0,

        # 옵티마이저 설정 (OptimArgs)
        "optim": {
            "lr": 0.0002,
            "weight_decay": 0.1,
            "pct_start": 0.05,
        },

        # LoRA 설정 (LoraArgs) - 고품질
        "lora": {
            "enable": True,
            "rank": 32,
            "scaling": 2.0,
            "ft_embed": False,
        },

        # 손실 함수 가중치
        "first_codebook_weight_multiplier": 1.0,
        "text_padding_weight": 0.5,

        # 체크포인팅
        "do_ckpt": True,
        "save_adapters": True,
        "num_ckpt_keep": 5,

        # 평가
        "do_eval": True,
        "eval_freq": 1000,

        # 효율성
        "gradient_checkpointing": True,
        "param_dtype": "bfloat16",

        # 분산 학습 - MPI 사용 시 ddp 권장
        "distributed_backend": "ddp",

        # WandB
        "wandb": {
            "project": "k-moshi",
            "run_name": "k-moshi-v1",
            "offline": True,  # 프록시 문제 시 오프라인 모드
        },

        # 기타
        "seed": 42,
        "full_finetuning": False,
        "overwrite_run_dir": False,
    }

    config_path = config_dir / "korean_finetune_template.yaml"

    # 주석 포함하여 저장
    header = """# ═══════════════════════════════════════════════════════════════════════
# 한국어 Moshi (K-Moshi) 파인튜닝 설정 템플릿
# ═══════════════════════════════════════════════════════════════════════
#
# 이 설정을 사용하기 전에:
# 1. 경로를 실제 환경에 맞게 수정
# 2. annotate.py로 한국어 데이터 전처리 (--lang ko 옵션)
# 3. GPU 메모리에 맞게 batch_size, duration_sec 조절
#
# A100 80GB 권장 설정:
#   - duration_sec: 30, batch_size: 2
#   - distributed_backend: ddp (MPI 사용 시)
#
# V100 32GB 권장 설정:
#   - duration_sec: 10, batch_size: 1
#   - num_microbatches: 2 (gradient accumulation)
#
# 분산 학습:
#   - torchrun 사용 시: distributed_backend: fsdp
#   - mpirun 사용 시: distributed_backend: ddp
# ═══════════════════════════════════════════════════════════════════════

"""

    with open(config_path, "w") as f:
        f.write(header)
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    logger.info(f"📄 한국어 템플릿 생성: {config_path}")

    print("\n[한국어 파인튜닝 권장 사항]")
    print("-" * 50)
    print("""
  1. 데이터 전처리:
     python -m moshi_finetune.annotate \\
         --input_dir /path/to/korean_audio \\
         --output_dir /path/to/output \\
         --lang ko \\
         --device cuda

  2. 토크나이저 주의사항:
     - 기본 SentencePiece 토크나이저 사용 (vocab_size=32K 유지)
     - 한국어는 자모 분리로 대부분 처리 가능
     - 미등록 단어(UNK) 비율 모니터링 필요

  3. 학습 모니터링:
     - text_loss: Inner Monologue 예측 성능
     - audio_loss: 오디오 코드북 예측 성능
     - 둘 다 감소해야 정상적인 학습
""")

    return config_path


def main():
    """메인 실행 함수"""
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent
    os.chdir(project_root)

    logger.info(f"📍 작업 디렉토리: {os.getcwd()}")

    # 설정 디렉토리 생성
    config_dir = Path("configs/tutorials")
    config_dir.mkdir(parents=True, exist_ok=True)

    # TrainArgs 설명
    explain_train_args()

    # LoRA 설명
    explain_lora()

    # 튜토리얼 설정 생성
    toy_config = create_toy_config(config_dir)

    # 한국어 템플릿 생성
    korean_config = create_korean_config_template(config_dir)

    print_banner("Step 3 완료!")
    logger.info("✅ 설정 파일 생성 완료")
    logger.info(f"   - 튜토리얼 설정: {toy_config}")
    logger.info(f"   - 한국어 템플릿: {korean_config}")
    logger.info("➡️  다음 단계: python scripts/tutorials/04_download_model.py")


if __name__ == "__main__":
    main()
