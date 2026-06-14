# W2v-BERT 2.0 Speaker Verification Model Download Guide

**작성일**: 2026-01-22
**버전**: 1.0

---

## 1. 개요

W2v-BERT 2.0 SV (Speaker Verification)는 VoxCeleb1-O 테스트 세트에서 **0.14% EER**를 달성한 SOTA 화자 인식 모델입니다. K-Moshi의 Zero-Shot Speaker Conditioning에서 화자 임베딩 추출에 사용됩니다.

### 1.1 모델 정보

| 항목 | 내용 |
|------|------|
| **논문** | [Enhancing Speaker Verification with w2v-BERT 2.0](https://arxiv.org/abs/2510.04213) |
| **HuggingFace** | https://huggingface.co/zl389/w2v-bert-2.0_SV |
| **GitHub** | https://github.com/BUTSpeechFIT/w2v-BERT-2.0_SV |
| **출력 차원** | 256-dim speaker embedding |
| **샘플링 레이트** | 16kHz |
| **성능** | VoxCeleb1-O: 0.14% EER (SOTA) |

### 1.2 모델 변형

| 모델 파일 | EER | 설명 |
|-----------|-----|------|
| `model_lmft_0.14.pth` | **0.14%** | LMFT (Layerwise Magnitude-Based Fine-Tuning) - **권장** |
| `model_vanilla_0.17.pth` | 0.17% | Vanilla fine-tuning |
| `model_student_0.26.pth` | 0.26% | Knowledge distillation compressed |

---

## 2. Linux GPU 서버 다운로드 가이드

### 2.1 Method 1: huggingface-cli 사용 (권장)

```bash
# 1. HuggingFace Hub 설치 (필요시)
pip install huggingface_hub

# 2. 모델 저장 디렉토리 생성
mkdir -p /path/to/model

# 3. HuggingFace에서 모델 다운로드
huggingface-cli download zl389/w2v-bert-2.0_SV \
    --local-dir /path/to/model \
    --local-dir-use-symlinks False
```

### 2.2 Method 2: Python 스크립트 사용

```python
#!/usr/bin/env python3
"""W2v-BERT 2.0 SV 모델 다운로드 스크립트"""

from huggingface_hub import hf_hub_download, snapshot_download
import os

# 저장 경로
MODEL_DIR = "/path/to/model"
os.makedirs(MODEL_DIR, exist_ok=True)

# 방법 1: 전체 저장소 다운로드
snapshot_download(
    repo_id="zl389/w2v-bert-2.0_SV",
    local_dir=MODEL_DIR,
    local_dir_use_symlinks=False,
)

# 방법 2: 특정 파일만 다운로드 (권장 모델)
model_path = hf_hub_download(
    repo_id="zl389/w2v-bert-2.0_SV",
    filename="model_lmft_0.14.pth",
    local_dir=MODEL_DIR,
    local_dir_use_symlinks=False,
)
print(f"Model downloaded to: {model_path}")
```

### 2.3 Method 3: wget 직접 다운로드

```bash
# 모델 저장 디렉토리 생성
mkdir -p /path/to/model
cd /path/to/model

# 권장 모델 다운로드 (0.14% EER)
wget https://huggingface.co/zl389/w2v-bert-2.0_SV/resolve/main/model_lmft_0.14.pth

# 또는 curl 사용
curl -L -O https://huggingface.co/zl389/w2v-bert-2.0_SV/resolve/main/model_lmft_0.14.pth
```

### 2.4 다운로드 확인

```bash
# 파일 확인
ls -la /path/to/model

# 예상 출력:
# -rw-r--r-- 1 user group 2.5G Jan 22 10:00 model_lmft_0.14.pth
# -rw-r--r-- 1 user group 2.5G Jan 22 10:00 model_vanilla_0.17.pth (optional)
# ...

# 파일 크기 확인 (약 2.5GB)
du -sh /path/to/model
```

---

## 3. 추가 의존성 설치

W2v-BERT 2.0 SV를 사용하려면 `transformers` 라이브러리가 필요합니다:

```bash
# transformers 설치 (w2v-bert-2.0 지원)
pip install transformers>=4.35.0

# 전체 의존성 확인
pip install torch torchaudio transformers huggingface_hub
```

### 3.1 w2v-BERT 2.0 베이스 모델 (선택사항)

Speaker encoder의 full forward pass를 위해 w2v-BERT 2.0 베이스 모델도 필요할 수 있습니다:

```bash
# 베이스 모델 다운로드 (자동 캐싱)
python -c "
from transformers import Wav2Vec2BertModel
model = Wav2Vec2BertModel.from_pretrained('facebook/w2v-bert-2.0')
print('w2v-bert-2.0 base model downloaded successfully')
"

# 또는 로컬에 명시적 저장
mkdir -p /path/to/model
huggingface-cli download facebook/w2v-bert-2.0 \
    --local-dir /path/to/model \
    --local-dir-use-symlinks False
```

---

## 4. K-Moshi 설정 예시

### 4.1 YAML 설정

```yaml
# example/korean_moshi_stage1_pretrain_spk.yaml

speaker:
  enabled: true
  method: "encoder"  # 또는 "both"

  encoder:
    encoder_type: "w2v_bert2"
    pretrained_path: "/path/to/model"
    output_dim: 256
    freeze: true
    sample_rate: 16000
    normalize_embedding: true

  conditioner:
    output_dim: 4096  # Moshi hidden dim
    initial_scale: 0.1
    use_layernorm: true
```

### 4.2 Python에서 직접 사용

```python
from finetune.modules.speaker_encoder import (
    SpeakerEncoderConfig,
    create_speaker_encoder,
)

# W2v-BERT 2.0 SV 설정
config = SpeakerEncoderConfig(
    encoder_type="w2v_bert2",
    pretrained_path="/path/to/model",
    output_dim=256,
    freeze=True,
    sample_rate=16000,
    normalize_embedding=True,
)

# Encoder 생성
encoder = create_speaker_encoder(config)
encoder = encoder.to("cuda")

# 사용 예시
import torch
reference_audio = torch.randn(1, 16000 * 5)  # 5초, 16kHz
reference_audio = reference_audio.to("cuda")

speaker_embedding = encoder(reference_audio)
print(f"Speaker embedding shape: {speaker_embedding.shape}")  # [1, 256]
```

---

## 5. 검증 스크립트

다운로드한 모델이 정상 작동하는지 확인:

```python
#!/usr/bin/env python3
"""W2v-BERT 2.0 SV 모델 검증 스크립트"""

import os
import sys
import torch

# 모델 경로
MODEL_PATH = "/path/to/model"

def verify_model():
    # 1. 파일 존재 확인
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return False
    print(f"✅ Model file exists: {MODEL_PATH}")

    # 2. 모델 로드 확인
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        print(f"✅ Model loaded successfully")

        # 체크포인트 구조 확인
        if "modules" in checkpoint:
            print(f"   - Contains 'modules' key")
            if "spk_model" in checkpoint["modules"]:
                print(f"   - Contains 'spk_model' state dict")

        # 키 샘플 출력
        if isinstance(checkpoint, dict):
            print(f"   - Top-level keys: {list(checkpoint.keys())[:5]}...")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

    # 3. K-Moshi speaker encoder 테스트
    try:
        sys.path.insert(0, "/path/to/workspace")
        from finetune.modules.speaker_encoder import (
            SpeakerEncoderConfig,
            create_speaker_encoder,
        )

        config = SpeakerEncoderConfig(
            encoder_type="w2v_bert2",
            pretrained_path=MODEL_PATH,
            output_dim=256,
            freeze=True,
        )

        encoder = create_speaker_encoder(config)
        print(f"✅ W2vBERT2SpeakerEncoder created successfully")

        # Forward pass 테스트
        test_audio = torch.randn(1, 16000 * 3)  # 3초
        embedding = encoder(test_audio)
        print(f"✅ Forward pass successful: {embedding.shape}")

        # 임베딩 정규화 확인
        norm = embedding.norm(dim=-1)
        print(f"   - Embedding norm: {norm.item():.4f} (should be ≈1.0 if normalized)")

    except ImportError as e:
        print(f"⚠️ K-Moshi modules not available: {e}")
        print("   Skipping integration test")
    except Exception as e:
        print(f"❌ Speaker encoder test failed: {e}")
        return False

    print("\n✅ All verification checks passed!")
    return True

if __name__ == "__main__":
    verify_model()
```

---

## 6. 문제 해결

### 6.1 다운로드 실패

```bash
# 프록시 설정이 필요한 경우
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080

# HuggingFace 토큰 인증 (private repo의 경우)
huggingface-cli login
```

### 6.2 메모리 부족

```python
# CPU에서 모델 로드 후 GPU로 이동
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
# ... 필요한 부분만 GPU로
```

### 6.3 transformers 버전 호환성

```bash
# 특정 버전 설치
pip install transformers==4.40.0

# 또는 최신 버전
pip install --upgrade transformers
```

---

## 7. 참조

- **논문**: Li et al., "Enhancing Speaker Verification with w2v-BERT 2.0 and Knowledge Distillation guided Structured Pruning" https://arxiv.org/abs/2510.04213
- **HuggingFace Repository**: https://huggingface.co/zl389/w2v-bert-2.0_SV
- **GitHub Repository**: https://github.com/BUTSpeechFIT/w2v-BERT-2.0_SV

---

*Last Updated: 2026-01-22*
*Author: K-Moshi Development Team*
