#!/usr/bin/env python3
"""
=============================================================================
Step 2: Explore Data Structure & Format
=============================================================================

목적: Moshi finetuning에 필요한 데이터 형식을 상세히 이해합니다.
      - 스테레오 WAV 구조 시각화
      - JSON 전사 파일 형식 분석
      - JSONL 메타데이터 형식 확인
      - Text-Audio 인터리빙 개념 이해

입력: data/daily-talk-contiguous/ (Step 1에서 다운로드)
출력: 콘솔 출력 + 시각화 (matplotlib 설치 시)

실행: python scripts/tutorials/02_explore_data.py
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


def analyze_jsonl_format(jsonl_path: Path):
    """JSONL 메타데이터 형식 분석"""
    print_banner("JSONL 메타데이터 형식 분석")

    logger.info(f"📂 파일: {jsonl_path}")

    with open(jsonl_path) as f:
        lines = f.readlines()

    logger.info(f"📊 총 {len(lines)}개의 데이터 엔트리")

    # 샘플 분석
    print("\n[JSONL 형식 설명]")
    print("-" * 50)
    print("각 줄은 하나의 대화 세션을 나타냅니다:")
    print('  {"path": "audio/file.wav", "duration": 45.32}')
    print("")
    print("필수 필드:")
    print("  - path: 스테레오 WAV 파일 경로")
    print("  - duration: 오디오 길이 (초)")

    print("\n[실제 데이터 샘플]")
    print("-" * 50)

    total_duration = 0
    for i, line in enumerate(lines[:5]):
        entry = json.loads(line)
        total_duration += entry.get("duration", 0)
        print(f"  #{i+1}: {json.dumps(entry, ensure_ascii=False)}")

    # 전체 통계
    all_durations = [json.loads(line).get("duration", 0) for line in lines]
    total = sum(all_durations)

    print(f"\n[데이터셋 통계]")
    print("-" * 50)
    print(f"  총 파일 수: {len(lines)}")
    print(f"  총 시간: {total:.1f}초 ({total/60:.1f}분)")
    print(f"  평균 길이: {total/len(lines):.1f}초")
    print(f"  최소 길이: {min(all_durations):.1f}초")
    print(f"  최대 길이: {max(all_durations):.1f}초")


def analyze_json_transcript(data_dir: Path):
    """JSON 전사 파일 형식 분석"""
    print_banner("JSON 전사 (Alignment) 형식 분석")

    # JSON 파일 찾기
    json_files = list(data_dir.rglob("*.json"))
    json_files = [f for f in json_files if f.suffix == ".json"]

    if not json_files:
        logger.warning("JSON 전사 파일을 찾을 수 없습니다.")
        return

    sample_json = json_files[0]
    logger.info(f"📂 파일: {sample_json}")

    with open(sample_json) as f:
        data = json.load(f)

    print("\n[JSON 전사 형식 설명]")
    print("-" * 50)
    print('"alignments" 배열의 각 요소:')
    print('  [텍스트, [시작시간, 종료시간], 화자]')
    print("")
    print("화자 라벨:")
    print("  - SPEAKER_MAIN: AI/Moshi (Left 채널)")
    print("  - SPEAKER_USER: 사용자 (Right 채널)")

    if "alignments" in data:
        alignments = data["alignments"]
        print(f"\n[실제 데이터 샘플] - 총 {len(alignments)}개 단어")
        print("-" * 50)

        # 처음 10개 표시
        for i, align in enumerate(alignments[:10]):
            word, (start, end), speaker = align
            duration = end - start
            print(f"  {start:6.2f}s - {end:6.2f}s ({duration:.2f}s): "
                  f"[{speaker.replace('SPEAKER_', '')}] \"{word}\"")

        if len(alignments) > 10:
            print(f"  ... 그 외 {len(alignments) - 10}개")

        # 화자별 통계
        speakers = {}
        for word, (start, end), speaker in alignments:
            if speaker not in speakers:
                speakers[speaker] = {"count": 0, "duration": 0}
            speakers[speaker]["count"] += 1
            speakers[speaker]["duration"] += end - start

        print(f"\n[화자별 통계]")
        print("-" * 50)
        for speaker, stats in speakers.items():
            print(f"  {speaker}:")
            print(f"    단어 수: {stats['count']}")
            print(f"    발화 시간: {stats['duration']:.1f}초")


def analyze_wav_structure(data_dir: Path):
    """스테레오 WAV 파일 구조 분석"""
    print_banner("스테레오 WAV 파일 구조 분석")

    wav_files = list(data_dir.rglob("*.wav"))

    if not wav_files:
        logger.warning("WAV 파일을 찾을 수 없습니다.")
        return

    sample_wav = wav_files[0]
    logger.info(f"📂 파일: {sample_wav}")

    try:
        import sphn
        import numpy as np

        # WAV 정보 읽기
        audio, sr = sphn.read(str(sample_wav))

        print("\n[WAV 파일 구조]")
        print("-" * 50)
        print(f"  파일명: {sample_wav.name}")
        print(f"  샘플레이트: {sr} Hz")
        print(f"  채널 수: {audio.shape[0]} (스테레오)")
        print(f"  샘플 수: {audio.shape[1]}")
        print(f"  길이: {audio.shape[1]/sr:.2f}초")

        print("\n[채널 구성]")
        print("-" * 50)
        print("  Channel 0 (Left):  SPEAKER_MAIN (AI/Moshi)")
        print("  Channel 1 (Right): SPEAKER_USER (사용자)")

        # 각 채널 RMS 분석
        rms_left = np.sqrt(np.mean(audio[0] ** 2))
        rms_right = np.sqrt(np.mean(audio[1] ** 2))

        print(f"\n[채널별 에너지 (RMS)]")
        print("-" * 50)
        print(f"  Left (MAIN):  {rms_left:.4f}")
        print(f"  Right (USER): {rms_right:.4f}")

        # 시각화 시도
        try:
            visualize_waveform(audio, sr, sample_wav.name)
        except ImportError:
            logger.info("💡 matplotlib 설치 시 파형 시각화 가능: pip install matplotlib")

    except ImportError:
        logger.warning("sphn 라이브러리가 없어 WAV 분석을 건너뜁니다.")
        logger.info("설치: pip install sphn")


def visualize_waveform(audio, sr, filename):
    """파형 시각화"""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # 시간 축 생성
    time = np.arange(audio.shape[1]) / sr

    # 처음 10초만 표시
    max_time = min(10, audio.shape[1] / sr)
    max_samples = int(max_time * sr)

    # Left 채널 (MAIN)
    axes[0].plot(time[:max_samples], audio[0, :max_samples], color='blue', linewidth=0.5)
    axes[0].set_ylabel('Left (MAIN)')
    axes[0].set_title(f'Stereo Waveform: {filename} (첫 10초)')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-1, 1)

    # Right 채널 (USER)
    axes[1].plot(time[:max_samples], audio[1, :max_samples], color='orange', linewidth=0.5)
    axes[1].set_ylabel('Right (USER)')
    axes[1].set_xlabel('Time (seconds)')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-1, 1)

    plt.tight_layout()

    # 저장
    output_path = Path("data/waveform_visualization.png")
    plt.savefig(output_path, dpi=150)
    logger.info(f"📊 파형 시각화 저장: {output_path}")
    plt.close()


def explain_interleaving():
    """Text-Audio 인터리빙 개념 설명"""
    print_banner("Text-Audio 인터리빙 개념")

    print("""
[Moshi의 Multi-Stream 구조]
─────────────────────────────────────────────────────────────

Moshi는 9개의 스트림을 동시에 처리합니다:

  시간 →     t=0    t=1    t=2    t=3    t=4    ...
  ────────────────────────────────────────────────
  Text:    │ [안]  │ [녕]  │ [PAD] │ [하]  │ [세]  │   ← Inner Monologue
  Audio 0: │ [C0]  │ [C0]  │ [C0]  │ [C0]  │ [C0]  │   ← Semantic (x100 가중치)
  Audio 1: │ [C1]  │ [C1]  │ [C1]  │ [C1]  │ [C1]  │   ← Acoustic
  Audio 2: │ [C2]  │ [C2]  │ [C2]  │ [C2]  │ [C2]  │
    ...
  Audio 7: │ [C7]  │ [C7]  │ [C7]  │ [C7]  │ [C7]  │

[프레임 레이트]
─────────────────────────────────────────────────────────────
  - Mimi 코덱: 12.5 Hz (80ms per frame)
  - 1초 = 12.5 프레임
  - 100초 청크 = 1,250 프레임

[텍스트 정렬 방식]
─────────────────────────────────────────────────────────────
  단어 타임스탬프 기반으로 텍스트 토큰을 배치:

  "안녕" (0.0s - 0.5s) → 프레임 0-6에 토큰 배치
  "하세요" (0.6s - 1.2s) → 프레임 7-15에 토큰 배치

  빈 프레임은 padding 토큰으로 채움

[손실 함수]
─────────────────────────────────────────────────────────────
  total_loss = text_loss + audio_loss

  - text_loss: Inner Monologue 예측 (SPEAKER_MAIN만)
  - audio_loss: 8개 codebook 예측
    - first_codebook_weight = 100 (semantic 강조)
""")


def main():
    """메인 실행 함수"""
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent
    os.chdir(project_root)

    logger.info(f"📍 작업 디렉토리: {os.getcwd()}")

    data_dir = Path("data/daily-talk-contiguous")

    if not data_dir.exists():
        logger.error(f"❌ 데이터 디렉토리가 없습니다: {data_dir}")
        logger.info("➡️  먼저 실행: python scripts/tutorials/01_download_toy_data.py")
        sys.exit(1)

    # JSONL 분석
    jsonl_files = list(data_dir.glob("*.jsonl"))
    if jsonl_files:
        analyze_jsonl_format(jsonl_files[0])

    # JSON 전사 분석
    analyze_json_transcript(data_dir)

    # WAV 구조 분석
    analyze_wav_structure(data_dir)

    # 인터리빙 개념 설명
    explain_interleaving()

    print_banner("Step 2 완료!")
    logger.info("✅ 데이터 구조 분석 완료")
    logger.info("➡️  다음 단계: python scripts/tutorials/03_create_config.py")


if __name__ == "__main__":
    main()
