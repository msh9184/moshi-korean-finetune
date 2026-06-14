#!/usr/bin/env python3
"""
=============================================================================
Step 5: Train with MPI + DDP
=============================================================================

목적: MPI와 DDP를 사용하여 Moshi 파인튜닝을 실행합니다.
      - mpirun 사용법 설명 (기본)
      - torchrun 대안 설명
      - 학습 모니터링 방법
      - 트러블슈팅 가이드

입력: configs/tutorials/toy_finetune.yaml (Step 3에서 생성)
      models/ (Step 4에서 다운로드)
      data/daily-talk-contiguous/ (Step 1에서 다운로드)

출력: runs/tutorial_toy/ (체크포인트, 로그)

실행: python scripts/tutorials/05_train_single_gpu.py
      (실제 학습은 shell 스크립트로 실행)
=============================================================================
"""

import logging
import os
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


def explain_training_process():
    """학습 프로세스 설명"""
    print_banner("Moshi 학습 프로세스")

    print("""
[학습 파이프라인]
─────────────────────────────────────────────────────────────

  1. 데이터 로딩
     JSONL 읽기 → sphn으로 오디오 청킹 → Mimi 인코딩 → 토큰화

  2. Forward Pass
     ┌─────────────────────────────────────────────────────┐
     │ 입력: [Text₀, Audio₀, ..., Audio₇]ₜ                │
     │                    ↓                                │
     │              Moshi Transformer                      │
     │                    ↓                                │
     │ 출력: [Text₁, Audio₁, ..., Audio₈]ₜ₊₁ (예측)        │
     └─────────────────────────────────────────────────────┘

  3. Loss 계산
     - text_loss: Inner Monologue 예측 손실
     - audio_loss: 8개 codebook 예측 손실
       (first_codebook_weight=100으로 semantic 강조)

  4. Backward + Optimizer Step
     - LoRA 파라미터만 업데이트
     - Gradient clipping (max_norm=1.0)

[분산 학습 설정: DDP + mpirun (권장)]
─────────────────────────────────────────────────────────────

  DDP (DistributedDataParallel)
  ─────────────────────────────────────────
  - 각 GPU에 전체 모델 복제
  - 설정이 간단하고 안정적
  - HPC 환경에서 MPI와 함께 사용 권장
  - LoRA 파인튜닝에 충분한 메모리 (A100 80GB)

     GPU 0           GPU 1           GPU 2           GPU 3
     ┌─────┐         ┌─────┐         ┌─────┐         ┌─────┐
     │Full │ ◄─────► │Full │ ◄─────► │Full │ ◄─────► │Full │
     │Model│         │Model│         │Model│         │Model│
     └─────┘         └─────┘         └─────┘         └─────┘

  설정 방법 (config.yaml):
     distributed_backend: ddp

[mpirun 런처 (기본값)]
─────────────────────────────────────────────────────────────

  단일 GPU:
    mpirun -np 1 python train.py --config <config.yaml>

  다중 GPU (단일 노드):
    mpirun -np 4 --npernode 4 \\
        -x CUDA_VISIBLE_DEVICES=0,1,2,3 \\
        -x MASTER_ADDR=$(hostname) \\
        -x MASTER_PORT=29500 \\
        python train.py --config <config.yaml>

  다중 노드 (hostfile 사용):
    mpirun -np 8 --npernode 4 -hostfile hostfile \\
        -x MASTER_ADDR=node1 -x MASTER_PORT=29500 \\
        python train.py --config <config.yaml>

[torchrun 런처 (대안)]
─────────────────────────────────────────────────────────────

  torchrun을 사용하려면 --torchrun 옵션 추가:
    bash scripts/tutorials/05_train_single_gpu.sh --torchrun

  단일 GPU:
    torchrun --nproc_per_node=1 train.py --config <config.yaml>
""")


def explain_monitoring():
    """학습 모니터링 설명"""
    print_banner("학습 모니터링")

    print("""
[로그 해석]
─────────────────────────────────────────────────────────────

  예시 로그:
    [Step 100/50000] loss=5.234 | lr=1.2e-4 | tokens/s=1234.5
                     | mem=45.2GB | time=0.8s/step

  주요 지표:
    - loss: 전체 손실 (text + audio)
            → 처음 ~6-7에서 시작, 점진적 감소 정상
    - lr: 현재 학습률
          → OneCycleLR로 증가→감소 패턴
    - tokens/s: 처리 속도
    - mem: GPU 메모리 사용량

[정상적인 학습 패턴]
─────────────────────────────────────────────────────────────

  Step     Loss      상태
  ─────────────────────────
  0-100    6.0-7.0   초기화
  100-1K   5.0-6.0   빠른 감소
  1K-10K   3.0-5.0   안정적 감소
  10K+     2.0-3.0   수렴 단계

[경고 신호]
─────────────────────────────────────────────────────────────

  ⚠️ Loss가 증가하거나 발산:
     → 학습률 낮추기 (lr / 2)

  ⚠️ Loss가 변하지 않음:
     → 학습률 높이기, 데이터 확인

  ⚠️ OOM (Out of Memory):
     → batch_size 줄이기, duration 줄이기

  ⚠️ NaN loss:
     → 학습률 확인, gradient clipping 확인
""")


def check_prerequisites():
    """학습 전 필수 요소 확인"""
    print_banner("학습 전 체크리스트")

    checks = []

    # 1. CUDA 확인
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            checks.append(("CUDA", True, f"{gpu_name} ({gpu_memory:.1f} GB)"))
        else:
            checks.append(("CUDA", False, "GPU not available"))
    except ImportError:
        checks.append(("PyTorch", False, "Not installed"))

    # 2. 설정 파일 확인
    config_path = Path("configs/tutorials/toy_finetune.yaml")
    checks.append(("Config", config_path.exists(), str(config_path)))

    # 3. 모델 파일 확인
    model_dir = Path("models/moshiko-pytorch-bf16")
    checks.append(("Moshi Model", model_dir.exists(), str(model_dir)))

    mimi_path = Path("models/tokenizer-e351c8d8-checkpoint125.safetensors")
    checks.append(("Mimi Codec", mimi_path.exists(), str(mimi_path)))

    tokenizer_path = Path("models/tokenizer_spm_32k_3.model")
    checks.append(("Tokenizer", tokenizer_path.exists(), str(tokenizer_path)))

    # 4. 데이터 확인
    data_dir = Path("data/daily-talk-contiguous")
    train_jsonl = Path("data/daily-talk-contiguous/train.jsonl")
    if train_jsonl.exists():
        checks.append(("Dataset", True, str(train_jsonl)))
    elif data_dir.exists():
        checks.append(("Dataset", True, f"{data_dir} (디렉토리 존재)"))
    else:
        checks.append(("Dataset", False, f"{data_dir} (다운로드 필요)"))

    # 결과 출력
    all_passed = True
    for name, passed, detail in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}: {detail}")
        if not passed:
            all_passed = False

    if not all_passed:
        print("\n⚠️  일부 필수 요소가 없습니다. 이전 단계를 먼저 실행해주세요.")
        print("\n필수 단계:")
        print("  1. python scripts/tutorials/01_download_toy_data.py  # 데이터 다운로드")
        print("  2. python scripts/tutorials/03_create_config.py      # 설정 파일 생성")
        print("  3. python scripts/tutorials/04_download_model.py     # 모델 다운로드")
        return False

    print("\n✅ 모든 필수 요소가 준비되었습니다!")
    return True


def show_training_command():
    """학습 명령어 표시"""
    print_banner("학습 실행 명령어")

    print("""
[mpirun 사용 (기본값, DDP 백엔드)]
─────────────────────────────────────────────────────────────

  # 튜토리얼 스크립트 사용 (권장)
  bash scripts/tutorials/05_train_single_gpu.sh

  # 또는 train_mpi.sh 사용
  bash scripts/train_mpi.sh --config configs/tutorials/toy_finetune.yaml --gpus 1

  # 직접 mpirun 사용
  mpirun -np 1 python train.py --config configs/tutorials/toy_finetune.yaml

[다중 GPU 학습 - mpirun 사용 (4 GPU)]
─────────────────────────────────────────────────────────────

  bash scripts/train_mpi.sh --config configs/tutorials/toy_finetune.yaml --gpus 4

  # 또는 직접 mpirun 사용
  mpirun -np 4 --npernode 4 \\
      -x CUDA_VISIBLE_DEVICES=0,1,2,3 \\
      -x MASTER_ADDR=$(hostname) \\
      -x MASTER_PORT=29500 \\
      python train.py --config configs/tutorials/toy_finetune.yaml

[torchrun 사용 (대안)]
─────────────────────────────────────────────────────────────

  # --torchrun 옵션 추가
  bash scripts/tutorials/05_train_single_gpu.sh --torchrun

  # 또는 직접 torchrun 사용
  torchrun --nproc_per_node=1 train.py \\
      --config configs/tutorials/toy_finetune.yaml

[백그라운드 실행 (긴 학습)]
─────────────────────────────────────────────────────────────

  # nohup 사용
  nohup bash scripts/train_mpi.sh \\
      --config configs/tutorials/toy_finetune.yaml \\
      > logs/train.log 2>&1 &

  # 로그 모니터링
  tail -f logs/train.log

  # tmux 사용 (권장)
  tmux new -s train
  bash scripts/train_mpi.sh --config configs/tutorials/toy_finetune.yaml
  # Ctrl+b, d 로 detach
  # tmux attach -t train 으로 재접속
""")


def show_troubleshooting():
    """트러블슈팅 가이드"""
    print_banner("트러블슈팅")

    print("""
[자주 발생하는 문제]
─────────────────────────────────────────────────────────────

  1. 데이터 파일 없음 ⭐
     ─────────────────────
     오류: FileNotFoundError: data/daily-talk-contiguous/train.jsonl

     해결:
     - Step 1 실행: python scripts/tutorials/01_download_toy_data.py
     - 프록시 문제 시 환경변수 설정:
       export HTTP_PROXY=http://your-proxy:port
       export HTTPS_PROXY=http://your-proxy:port

  2. CUDA Out of Memory
     ─────────────────────
     오류: RuntimeError: CUDA out of memory

     해결:
     - batch_size 줄이기 (2 → 1)
     - duration_sec 줄이기 (30 → 10)
     - lora.rank 줄이기 (32 → 8)

  3. MPI 관련 오류
     ─────────────────────
     오류: mpirun command not found

     해결:
     - OpenMPI 설치 확인: which mpirun
     - 또는 torchrun 사용: --torchrun 옵션 추가

  4. NCCL 통신 오류
     ─────────────────────
     오류: NCCL timeout 또는 connection refused

     해결:
     - 방화벽 포트 열기
     - NCCL_DEBUG=INFO 환경변수로 디버깅
     - 단일 GPU로 먼저 테스트

[유용한 환경변수]
─────────────────────────────────────────────────────────────

  # CUDA 메모리 최적화
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  # NCCL 디버깅
  export NCCL_DEBUG=INFO

  # 프록시 (HuggingFace 다운로드 시)
  export HTTP_PROXY=http://proxy:port
  export HTTPS_PROXY=http://proxy:port
""")


def main():
    """메인 실행 함수"""
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent
    os.chdir(project_root)

    logger.info(f"📍 작업 디렉토리: {os.getcwd()}")

    # 학습 프로세스 설명
    explain_training_process()

    # 모니터링 설명
    explain_monitoring()

    # 필수 요소 확인
    ready = check_prerequisites()

    # 학습 명령어
    show_training_command()

    # 트러블슈팅
    show_troubleshooting()

    print_banner("Step 5 설명 완료!")

    if ready:
        logger.info("✅ 학습 준비 완료")
        logger.info("")
        logger.info("📌 실제 학습을 시작하려면:")
        logger.info("   bash scripts/tutorials/05_train_single_gpu.sh")
        logger.info("")
        logger.info("📌 또는 train_mpi.sh 사용:")
        logger.info("   bash scripts/train_mpi.sh --config configs/tutorials/toy_finetune.yaml")
        logger.info("")
        logger.info("➡️  학습 완료 후: bash scripts/tutorials/06_inference.sh")
    else:
        logger.warning("⚠️  먼저 이전 단계를 완료해주세요.")


if __name__ == "__main__":
    main()
