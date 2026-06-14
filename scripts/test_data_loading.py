#!/usr/bin/env python3
"""
=============================================================================
데이터 로딩 테스트 스크립트
=============================================================================

목적: 학습 전에 데이터 로딩이 정상적으로 동작하는지 확인합니다.
      - JSONL 파일 경로 해석
      - JSON alignment 파일 로딩
      - sphn 라이브러리 동작 확인
      - 토크나이저 동작 확인

실행: python scripts/test_data_loading.py --config configs/tutorials/toy_finetune.yaml

=============================================================================
"""

import argparse
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


def test_jsonl_paths(jsonl_path: Path) -> dict:
    """JSONL 파일의 경로가 올바른지 확인"""
    result = {
        "jsonl_path": str(jsonl_path),
        "jsonl_exists": jsonl_path.exists(),
        "entries": 0,
        "wav_found": 0,
        "wav_missing": 0,
        "json_found": 0,
        "json_missing": 0,
        "errors": []
    }

    if not result["jsonl_exists"]:
        result["errors"].append(f"JSONL file not found: {jsonl_path}")
        return result

    jsonl_dir = jsonl_path.parent

    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            result["entries"] += 1
            try:
                data = json.loads(line.strip())
                rel_path = data.get("path", "")

                # WAV 파일 확인
                wav_path = jsonl_dir / rel_path
                if wav_path.exists():
                    result["wav_found"] += 1

                    # JSON alignment 파일 확인
                    json_path = wav_path.with_suffix('.json')
                    if json_path.exists():
                        result["json_found"] += 1
                    else:
                        result["json_missing"] += 1
                        if len(result["errors"]) < 5:
                            result["errors"].append(f"Missing JSON: {json_path}")
                else:
                    result["wav_missing"] += 1
                    if len(result["errors"]) < 5:
                        result["errors"].append(f"Missing WAV: {wav_path}")

            except json.JSONDecodeError as e:
                result["errors"].append(f"Line {i}: JSON decode error")

            if i >= 99:  # 처음 100개만 확인
                break

    return result


def test_sphn_loading(jsonl_path: Path, num_samples: int = 3) -> dict:
    """sphn 라이브러리로 데이터 로딩 테스트"""
    result = {
        "sphn_available": False,
        "samples_loaded": 0,
        "sample_paths": [],
        "errors": []
    }

    try:
        import sphn
        result["sphn_available"] = True
    except ImportError:
        result["errors"].append("sphn library not installed")
        return result

    try:
        dataset = sphn.dataset_jsonl(
            str(jsonl_path),
            duration_sec=10.0,
            num_threads=1,
            sample_rate=24000,
            pad_last_segment=True,
        )

        for i, sample in enumerate(dataset.seq()):
            if i >= num_samples:
                break

            result["samples_loaded"] += 1
            sample_path = sample.get("path", "UNKNOWN")
            result["sample_paths"].append(sample_path)

            # 경로가 절대 경로인지 상대 경로인지 확인
            if i == 0:
                if os.path.isabs(sample_path):
                    result["path_type"] = "absolute"
                else:
                    result["path_type"] = "relative"

    except Exception as e:
        result["errors"].append(f"sphn loading error: {e}")

    return result


def test_json_alignment(jsonl_path: Path) -> dict:
    """JSON alignment 파일 포맷 테스트"""
    result = {
        "alignments_checked": 0,
        "valid_alignments": 0,
        "sample_alignment": None,
        "errors": []
    }

    jsonl_dir = jsonl_path.parent

    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:  # 처음 3개만 확인
                break

            try:
                data = json.loads(line.strip())
                rel_path = data.get("path", "")
                json_path = jsonl_dir / Path(rel_path).with_suffix('.json')

                if json_path.exists():
                    result["alignments_checked"] += 1

                    with open(json_path, 'r') as jf:
                        align_data = json.load(jf)

                    if "alignments" in align_data:
                        alignments = align_data["alignments"]
                        if len(alignments) > 0:
                            first = alignments[0]
                            # 포맷 확인: [word, [start, end], speaker]
                            if (isinstance(first, list) and len(first) >= 3 and
                                isinstance(first[1], list) and len(first[1]) == 2):
                                result["valid_alignments"] += 1
                                if result["sample_alignment"] is None:
                                    result["sample_alignment"] = {
                                        "file": str(json_path.name),
                                        "num_words": len(alignments),
                                        "first_word": first[0],
                                        "time_range": first[1],
                                        "speaker": first[2]
                                    }
                            else:
                                result["errors"].append(f"Invalid alignment format in {json_path}")
                    else:
                        result["errors"].append(f"No 'alignments' key in {json_path}")

            except Exception as e:
                result["errors"].append(f"Error checking alignment: {e}")

    return result


def test_full_pipeline(config_path: str) -> dict:
    """전체 파이프라인 테스트 (모델 로딩 없이)"""
    result = {
        "config_loaded": False,
        "data_path": None,
        "errors": []
    }

    try:
        from finetune.args import TrainArgs
        args = TrainArgs.load(config_path, drop_extra_fields=False)
        result["config_loaded"] = True
        result["data_path"] = args.data.train_data
        result["duration_sec"] = args.duration_sec
    except Exception as e:
        result["errors"].append(f"Config loading error: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description="데이터 로딩 테스트")
    parser.add_argument("--config", type=str, default="configs/tutorials/toy_finetune.yaml",
                        help="설정 파일 경로")
    parser.add_argument("--jsonl", type=str, default=None,
                        help="JSONL 파일 경로 (config 대신 직접 지정)")
    args = parser.parse_args()

    print_banner("데이터 로딩 테스트")

    # 1. JSONL 파일 경로 결정
    if args.jsonl:
        jsonl_path = Path(args.jsonl)
    else:
        # config에서 경로 추출
        config_result = test_full_pipeline(args.config)
        if config_result["config_loaded"]:
            jsonl_path = Path(config_result["data_path"])
            print(f"📁 Config에서 데이터 경로 추출: {jsonl_path}")
            print(f"   duration_sec: {config_result['duration_sec']}")
        else:
            print("❌ Config 로딩 실패")
            for err in config_result["errors"]:
                print(f"   - {err}")
            # 기본 경로 시도
            jsonl_path = Path("data/daily-talk-contiguous/dailytalk.jsonl")
            print(f"   기본 경로 사용: {jsonl_path}")

    # 2. JSONL 경로 검증
    print_banner("Step 1: JSONL 경로 검증")
    path_result = test_jsonl_paths(jsonl_path)

    print(f"JSONL 파일: {path_result['jsonl_path']}")
    print(f"  존재 여부: {'✅' if path_result['jsonl_exists'] else '❌'}")

    if path_result['jsonl_exists']:
        print(f"  총 엔트리 (처음 100개 검사): {path_result['entries']}")
        print(f"  WAV 발견: {path_result['wav_found']}")
        print(f"  WAV 누락: {path_result['wav_missing']}")
        print(f"  JSON 발견: {path_result['json_found']}")
        print(f"  JSON 누락: {path_result['json_missing']}")

    if path_result['errors']:
        print(f"\n  ⚠️ 에러:")
        for err in path_result['errors'][:5]:
            print(f"     - {err}")

    # 3. sphn 로딩 테스트
    print_banner("Step 2: sphn 라이브러리 테스트")
    sphn_result = test_sphn_loading(jsonl_path)

    print(f"sphn 사용 가능: {'✅' if sphn_result['sphn_available'] else '❌'}")
    if sphn_result['sphn_available']:
        print(f"로드된 샘플 수: {sphn_result['samples_loaded']}")
        if 'path_type' in sphn_result:
            print(f"경로 타입: {sphn_result['path_type']}")
        print(f"샘플 경로:")
        for p in sphn_result['sample_paths']:
            print(f"  - {p}")

    if sphn_result['errors']:
        print(f"\n⚠️ 에러:")
        for err in sphn_result['errors']:
            print(f"   - {err}")

    # 4. JSON alignment 테스트
    print_banner("Step 3: JSON Alignment 포맷 테스트")
    align_result = test_json_alignment(jsonl_path)

    print(f"확인된 alignment 파일: {align_result['alignments_checked']}")
    print(f"유효한 alignment: {align_result['valid_alignments']}")

    if align_result['sample_alignment']:
        sample = align_result['sample_alignment']
        print(f"\n샘플 alignment:")
        print(f"  파일: {sample['file']}")
        print(f"  단어 수: {sample['num_words']}")
        print(f"  첫 단어: '{sample['first_word']}'")
        print(f"  시간 범위: {sample['time_range']}")
        print(f"  화자: {sample['speaker']}")

    if align_result['errors']:
        print(f"\n⚠️ 에러:")
        for err in align_result['errors']:
            print(f"   - {err}")

    # 5. 최종 결과
    print_banner("테스트 결과 요약")

    all_pass = (
        path_result['jsonl_exists'] and
        path_result['wav_missing'] == 0 and
        path_result['json_missing'] == 0 and
        sphn_result['sphn_available'] and
        sphn_result['samples_loaded'] > 0 and
        align_result['valid_alignments'] > 0
    )

    if all_pass:
        print("✅ 모든 테스트 통과! 학습을 시작할 수 있습니다.")
        print("\n다음 명령으로 학습을 시작하세요:")
        print(f"  bash scripts/train_mpi.sh --config {args.config} --gpus 1")
    else:
        print("❌ 일부 테스트 실패. 위의 에러를 확인하세요.")

        if path_result['wav_missing'] > 0 or path_result['json_missing'] > 0:
            print("\n💡 JSONL 파일의 경로가 올바른지 확인하세요.")
            print("   JSONL 파일 위치 기준으로 상대 경로가 올바르게 설정되어야 합니다.")

        if not sphn_result['sphn_available']:
            print("\n💡 sphn 라이브러리를 설치하세요:")
            print("   pip install sphn")


if __name__ == "__main__":
    main()
