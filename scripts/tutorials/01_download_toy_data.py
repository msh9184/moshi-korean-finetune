#!/usr/bin/env python3
"""
=============================================================================
Step 1: Download Toy Dataset (DailyTalkContiguous)
=============================================================================

목적: Moshi finetuning을 위한 toy 데이터셋을 다운로드합니다.
      DailyTalkContiguous는 Kyutai에서 제공하는 스테레오 대화 데이터셋입니다.

입력: 없음 (HuggingFace에서 자동 다운로드)
출력: data/daily-talk-contiguous/ 디렉토리에 데이터셋 저장

실행: python scripts/tutorials/01_download_toy_data.py
      또는
      bash scripts/tutorials/01_download_toy_data.sh
=============================================================================
"""

import logging
import os
import sys
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def print_banner(title: str):
    """배너 출력"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def download_dataset(data_dir: str = "data/daily-talk-contiguous"):
    """
    DailyTalkContiguous 데이터셋 다운로드

    이 데이터셋은:
    - 스테레오 WAV 파일 (Left=Speaker A, Right=Speaker B)
    - 이미 단어 수준 타임스탬프가 포함된 JSON 파일
    - JSONL 메타데이터 파일
    """
    print_banner("Step 1: Download Toy Dataset")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub이 설치되지 않았습니다.")
        logger.info("설치 명령: pip install huggingface_hub")
        sys.exit(1)

    # 데이터 디렉토리 생성
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"📂 데이터 저장 경로: {data_path.absolute()}")
    logger.info("📥 HuggingFace에서 DailyTalkContiguous 데이터셋 다운로드 중...")
    logger.info("   (처음 실행 시 약 100MB 다운로드)")

    try:
        local_dir = snapshot_download(
            "kyutai/DailyTalkContiguous",
            repo_type="dataset",
            local_dir=str(data_path),
        )
        logger.info(f"✅ 다운로드 완료: {local_dir}")
    except Exception as e:
        logger.error(f"❌ 다운로드 실패: {e}")
        logger.info("💡 프록시 문제일 경우 환경변수 설정이 필요할 수 있습니다:")
        logger.info("   export HTTP_PROXY=http://your-proxy:port")
        logger.info("   export HTTPS_PROXY=http://your-proxy:port")
        sys.exit(1)

    return local_dir


def explore_downloaded_data(data_dir: str):
    """다운로드된 데이터 구조 확인"""
    print_banner("Downloaded Data Structure")

    data_path = Path(data_dir)

    # 파일 목록 출력
    logger.info("📁 다운로드된 파일 구조:")

    all_files = list(data_path.rglob("*"))
    files_by_type = {}

    for f in all_files:
        if f.is_file():
            ext = f.suffix.lower()
            if ext not in files_by_type:
                files_by_type[ext] = []
            files_by_type[ext].append(f)

    for ext, files in sorted(files_by_type.items()):
        print(f"\n  {ext if ext else '(no ext)'}: {len(files)} 파일")
        for f in files[:3]:  # 각 타입별 최대 3개만 표시
            size_kb = f.stat().st_size / 1024
            print(f"    - {f.name} ({size_kb:.1f} KB)")
        if len(files) > 3:
            print(f"    ... 그 외 {len(files) - 3}개")

    # JSONL 파일 확인
    jsonl_files = list(data_path.glob("*.jsonl"))
    if jsonl_files:
        print("\n" + "-" * 50)
        logger.info("📋 JSONL 메타데이터 파일 내용 샘플:")

        import json
        with open(jsonl_files[0]) as f:
            lines = f.readlines()[:3]
            for i, line in enumerate(lines):
                entry = json.loads(line)
                print(f"\n  Entry {i+1}:")
                for k, v in entry.items():
                    print(f"    {k}: {v}")

    # WAV 파일 정보
    wav_files = list(data_path.rglob("*.wav"))
    if wav_files:
        print("\n" + "-" * 50)
        logger.info("🎵 WAV 파일 정보 샘플:")

        try:
            import sphn
            sample_wav = wav_files[0]
            duration = sphn.duration(str(sample_wav))
            print(f"\n  파일: {sample_wav.name}")
            print(f"  길이: {duration:.2f}초")
        except ImportError:
            logger.warning("sphn 라이브러리가 없어 WAV 정보를 표시할 수 없습니다.")

    # JSON 전사 파일 확인
    json_files = [f for f in data_path.rglob("*.json") if f.suffix == ".json"]
    if json_files:
        print("\n" + "-" * 50)
        logger.info("📝 JSON 전사 파일 내용 샘플:")

        import json
        sample_json = json_files[0]
        with open(sample_json) as f:
            data = json.load(f)

        print(f"\n  파일: {sample_json.name}")
        if "alignments" in data:
            print(f"  alignments 개수: {len(data['alignments'])}")
            print("\n  처음 5개 alignment:")
            for align in data["alignments"][:5]:
                print(f"    {align}")

    print("\n" + "=" * 70)


def main():
    """메인 실행 함수"""
    # 프로젝트 루트로 이동
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent
    os.chdir(project_root)

    logger.info(f"📍 작업 디렉토리: {os.getcwd()}")

    # 데이터 다운로드
    data_dir = "data/daily-talk-contiguous"
    download_dataset(data_dir)

    # 데이터 구조 확인
    explore_downloaded_data(data_dir)

    print_banner("Step 1 완료!")
    logger.info("✅ Toy 데이터셋 다운로드 및 확인 완료")
    logger.info("➡️  다음 단계: python scripts/tutorials/02_explore_data.py")


if __name__ == "__main__":
    main()
