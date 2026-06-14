#!/usr/bin/env python3
"""
=============================================================================
데이터 경로 수정 스크립트
=============================================================================

목적: JSONL 파일의 상대 경로를 절대 경로로 변환하고,
      alignment JSON 파일 포맷을 검증합니다.

문제: interleaver.py에서 sample["path"]를 그대로 사용하여 JSON 파일을 열려고 함
      - JSONL: {"path": "data_stereo/0.wav", ...}
      - 실제 위치: data/daily-talk-contiguous/data_stereo/0.wav
      - interleaver.py가 열려는 경로: data_stereo/0.json (실패!)

해결: JSONL 파일의 path를 JSONL 파일 기준 상대경로 또는 절대경로로 수정

실행: python scripts/fix_data_paths.py [--check-only] [--fix]

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


def check_alignment_format(json_path: Path) -> dict:
    """Alignment JSON 파일 포맷 검증"""
    result = {
        "path": str(json_path),
        "exists": json_path.exists(),
        "valid_json": False,
        "has_alignments": False,
        "alignment_format_ok": False,
        "num_alignments": 0,
        "sample": None,
        "error": None
    }

    if not result["exists"]:
        result["error"] = "File does not exist"
        return result

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        result["valid_json"] = True

        if "alignments" in data:
            result["has_alignments"] = True
            alignments = data["alignments"]
            result["num_alignments"] = len(alignments)

            if len(alignments) > 0:
                # 첫 번째 alignment 검사
                first_align = alignments[0]

                # 예상 포맷: [word, [start, end], speaker]
                # 또는: {"word": "...", "start": 0.0, "end": 0.5, "speaker": "..."}
                if isinstance(first_align, list) and len(first_align) >= 3:
                    word, ts, speaker = first_align[0], first_align[1], first_align[2]
                    if isinstance(ts, (list, tuple)) and len(ts) == 2:
                        result["alignment_format_ok"] = True
                        result["sample"] = {
                            "word": word,
                            "start": ts[0],
                            "end": ts[1],
                            "speaker": speaker
                        }
                elif isinstance(first_align, dict):
                    # 딕셔너리 형식인 경우
                    result["sample"] = first_align
                    result["alignment_format_ok"] = False
                    result["error"] = "Alignment is dict format, expected list format"
                else:
                    result["error"] = f"Unknown alignment format: {type(first_align)}"
        else:
            result["error"] = "No 'alignments' key found in JSON"
            # 어떤 키가 있는지 확인
            result["available_keys"] = list(data.keys())[:5]

    except json.JSONDecodeError as e:
        result["error"] = f"JSON decode error: {e}"
    except Exception as e:
        result["error"] = f"Error: {e}"

    return result


def fix_jsonl_paths(jsonl_path: Path, dry_run: bool = True) -> dict:
    """JSONL 파일의 경로를 수정"""
    result = {
        "jsonl_path": str(jsonl_path),
        "total_lines": 0,
        "fixed_lines": 0,
        "wav_found": 0,
        "wav_missing": 0,
        "json_found": 0,
        "json_missing": 0,
        "errors": []
    }

    jsonl_dir = jsonl_path.parent

    # 원본 읽기
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    result["total_lines"] = len(lines)

    fixed_lines = []

    for i, line in enumerate(lines):
        try:
            data = json.loads(line.strip())
            original_path = data.get("path", "")

            # 현재 경로 확인
            current_wav = jsonl_dir / original_path

            if current_wav.exists():
                result["wav_found"] += 1

                # JSON alignment 파일 확인
                json_file = current_wav.with_suffix('.json')
                if json_file.exists():
                    result["json_found"] += 1
                else:
                    result["json_missing"] += 1
                    if len(result["errors"]) < 5:
                        result["errors"].append(f"Missing JSON: {json_file}")

                # 경로를 JSONL 기준 상대경로로 유지 (sphn이 처리함)
                # 하지만 절대경로로 변환하면 더 안전함
                # 여기서는 JSONL 디렉토리 기준 경로가 맞는지 확인

                fixed_lines.append(line)

            else:
                # 경로가 잘못된 경우 - 실제 파일 찾기
                wav_name = Path(original_path).name

                # data_stereo 하위에서 찾기
                possible_paths = list(jsonl_dir.rglob(wav_name))

                if possible_paths:
                    correct_path = possible_paths[0].relative_to(jsonl_dir)
                    data["path"] = str(correct_path)
                    fixed_lines.append(json.dumps(data, ensure_ascii=False) + "\n")
                    result["fixed_lines"] += 1
                    result["wav_found"] += 1

                    # JSON 확인
                    json_file = possible_paths[0].with_suffix('.json')
                    if json_file.exists():
                        result["json_found"] += 1
                    else:
                        result["json_missing"] += 1
                else:
                    result["wav_missing"] += 1
                    if len(result["errors"]) < 5:
                        result["errors"].append(f"WAV not found: {original_path}")
                    fixed_lines.append(line)

        except json.JSONDecodeError as e:
            result["errors"].append(f"Line {i}: JSON decode error")
            fixed_lines.append(line)

    # 백업 및 저장
    if not dry_run and result["fixed_lines"] > 0:
        backup_path = jsonl_path.with_suffix('.jsonl.bak')
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        logger.info(f"Backup saved to: {backup_path}")

        with open(jsonl_path, 'w', encoding='utf-8') as f:
            f.writelines(fixed_lines)
        logger.info(f"Fixed JSONL saved to: {jsonl_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="데이터 경로 수정 및 검증")
    parser.add_argument("--data-dir", type=str, default="data/daily-talk-contiguous",
                        help="데이터 디렉토리 경로")
    parser.add_argument("--check-only", action="store_true",
                        help="검사만 수행 (수정하지 않음)")
    parser.add_argument("--fix", action="store_true",
                        help="문제를 수정")
    parser.add_argument("--check-alignments", type=int, default=3,
                        help="검사할 alignment 파일 수")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        logger.error(f"데이터 디렉토리가 존재하지 않습니다: {data_dir}")
        sys.exit(1)

    print_banner("데이터 경로 검증 및 수정")

    # 1. JSONL 파일 찾기
    jsonl_files = list(data_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.error(f"JSONL 파일을 찾을 수 없습니다: {data_dir}")
        sys.exit(1)

    logger.info(f"발견된 JSONL 파일: {len(jsonl_files)}")
    for jsonl_file in jsonl_files:
        logger.info(f"  - {jsonl_file}")

    # 2. JSONL 경로 검증 및 수정
    print_banner("JSONL 경로 검증")

    for jsonl_file in jsonl_files:
        result = fix_jsonl_paths(jsonl_file, dry_run=not args.fix)

        print(f"\n📁 {jsonl_file.name}")
        print(f"   총 라인: {result['total_lines']}")
        print(f"   WAV 발견: {result['wav_found']}")
        print(f"   WAV 누락: {result['wav_missing']}")
        print(f"   JSON 발견: {result['json_found']}")
        print(f"   JSON 누락: {result['json_missing']}")
        print(f"   수정 필요: {result['fixed_lines']}")

        if result['errors']:
            print(f"\n   ⚠️ 에러 (처음 5개):")
            for err in result['errors'][:5]:
                print(f"      - {err}")

    # 3. Alignment JSON 포맷 검증
    print_banner("Alignment JSON 포맷 검증")

    json_files = list(data_dir.rglob("*.json"))
    if json_files:
        logger.info(f"발견된 JSON 파일: {len(json_files)}")

        # 샘플 검사
        sample_count = min(args.check_alignments, len(json_files))
        for json_file in json_files[:sample_count]:
            result = check_alignment_format(json_file)

            status = "✅" if result['alignment_format_ok'] else "❌"
            print(f"\n{status} {json_file.name}")
            print(f"   유효한 JSON: {result['valid_json']}")
            print(f"   alignments 키: {result['has_alignments']}")
            print(f"   포맷 정상: {result['alignment_format_ok']}")
            print(f"   alignment 수: {result['num_alignments']}")

            if result['sample']:
                print(f"   샘플: {result['sample']}")
            if result['error']:
                print(f"   ⚠️ 에러: {result['error']}")
            if 'available_keys' in result:
                print(f"   사용 가능한 키: {result['available_keys']}")
    else:
        logger.warning("JSON 파일을 찾을 수 없습니다!")

    # 4. 요약
    print_banner("요약 및 권장사항")

    if args.check_only:
        print("ℹ️  검사 모드로 실행됨 (--fix 옵션으로 수정 가능)")

    print("\n다음 단계:")
    print("1. 위 결과를 확인하고 문제가 있으면 --fix 옵션으로 수정")
    print("2. alignment 포맷이 올바르지 않으면 annotate.py로 재생성 필요")
    print("3. 수정 후 학습 재시도")


if __name__ == "__main__":
    main()
