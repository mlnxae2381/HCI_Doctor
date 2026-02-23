"""
X-ray LoRA 체크포인트를 최종 모델로 변환
Best checkpoint (step 3600)를 xray_lora_adapter로 저장
"""

import shutil
from pathlib import Path

# 경로 설정
RESULTS_DIR = Path("/home/students/cs/doctor/AI_Doctor/test/xray_lora_results")
BEST_CHECKPOINT = RESULTS_DIR / "checkpoints" / "checkpoint-3600"
FINAL_ADAPTER = RESULTS_DIR / "xray_lora_adapter"

print("=" * 60)
print("X-ray LoRA 최종 모델 생성")
print("=" * 60)

# 체크포인트 존재 확인
if not BEST_CHECKPOINT.exists():
    raise FileNotFoundError(f"Best checkpoint not found: {BEST_CHECKPOINT}")

print(f"\n✅ Best checkpoint 확인: {BEST_CHECKPOINT}")

# 기존 final adapter 삭제 (있다면)
if FINAL_ADAPTER.exists():
    print(f"\n⚠️  기존 adapter 삭제 중: {FINAL_ADAPTER}")
    shutil.rmtree(FINAL_ADAPTER)

# 체크포인트를 final adapter로 복사
print(f"\n📦 체크포인트를 최종 모델로 복사 중...")
shutil.copytree(BEST_CHECKPOINT, FINAL_ADAPTER)

print(f"✅ 복사 완료: {FINAL_ADAPTER}")

# 파일 목록 확인
files = list(FINAL_ADAPTER.glob("*"))
print(f"\n📁 최종 모델 파일 ({len(files)}개):")
for f in sorted(files):
    size = f.stat().st_size / (1024**2)  # MB
    print(f"  - {f.name} ({size:.2f} MB)")

print("\n" + "=" * 60)
print("✨ X-ray LoRA 최종 모델 생성 완료!")
print("=" * 60)
print(f"📁 최종 모델 경로: {FINAL_ADAPTER}")
print("\n🎯 다음 단계:")
print("  1. python test_medgemma_lora_xray.py  # 테스트 평가")
print("  2. 웹앱 백엔드에서 모델 사용 가능")
print("=" * 60)
