"""
X-ray LoRA 어댑터에 Processor 추가
테스트 및 배포를 위해 필요
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU만 사용 (빠른 로드)

from pathlib import Path
from transformers import AutoProcessor
from dotenv import load_dotenv

# .env 로드
load_dotenv()

# 경로 설정
ADAPTER_PATH = Path("/home/students/cs/doctor/AI_Doctor/test/xray_lora_results/xray_lora_adapter")
MODEL_ID = "google/medgemma-1.5-4b-it"

print("=" * 60)
print("X-ray LoRA 어댑터에 Processor 추가")
print("=" * 60)

# HuggingFace 로그인
from huggingface_hub import login
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN이 설정되지 않았습니다!")

print(f"\n✅ HuggingFace 로그인 중...")
login(token=HF_TOKEN)

# Processor 로드
print(f"\n📦 Processor 로드 중... ({MODEL_ID})")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# Processor 저장
print(f"\n💾 Processor 저장 중... ({ADAPTER_PATH})")
processor.save_pretrained(ADAPTER_PATH)

print(f"\n✅ Processor 저장 완료!")

# 최종 파일 목록
files = list(ADAPTER_PATH.glob("*"))
print(f"\n📁 최종 어댑터 파일 ({len(files)}개):")
for f in sorted(files):
    size = f.stat().st_size / (1024**2) if f.is_file() else 0  # MB
    print(f"  - {f.name} ({size:.2f} MB)")

print("\n" + "=" * 60)
print("✨ X-ray LoRA 어댑터 완성!")
print("=" * 60)
print(f"📁 어댑터 경로: {ADAPTER_PATH}")
print("\n🎯 이제 테스트 및 배포 가능합니다:")
print("  1. python test_medgemma_lora_xray.py")
print("  2. 웹앱에서 모델 사용")
print("=" * 60)
