"""
MedGemma CT-LoRA 테스트 스크립트
학습된 LoRA 어댑터를 불러와서 테스트 세트 평가
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU 0번만 사용

import json
import torch
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# .env 로드
load_dotenv()

# 경로 설정
BASE_DIR = Path("/home/students/cs/doctor/AI_Doctor/data")
DATA_DIR = BASE_DIR / "processed"
OUTPUT_DIR = Path("/home/students/cs/doctor/AI_Doctor/test/ct_lora_results")
ADAPTER_PATH = OUTPUT_DIR / "ct_lora_adapter"

MODEL_ID = "google/medgemma-1.5-4b-it"

# 클래스 매핑
LABEL_MAP = {"normal": 0, "benign": 1, "malignant": 2}
LABEL_NAMES = ["normal", "benign", "malignant"]

print("=" * 60)
print("MedGemma CT-LoRA 테스트 평가")
print("=" * 60)

# ============================================
# 1. 모델 및 어댑터 로드
# ============================================
print("\n[1단계] 학습된 모델 로드")

from huggingface_hub import login
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN이 설정되지 않았습니다!")

print(f"HuggingFace 로그인 중... (토큰: {HF_TOKEN[:10]}...)")
login(token=HF_TOKEN)

# Processor 로드
print("Processor 로드 중...")
processor = AutoProcessor.from_pretrained(ADAPTER_PATH, trust_remote_code=True)

# 베이스 모델 로드
print("베이스 모델 로드 중...")
base_model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# LoRA 어댑터 로드
print(f"LoRA 어댑터 로드 중... ({ADAPTER_PATH})")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model = model.to("cuda")
model.eval()

print("모델 로드 완료!")

# ============================================
# 2. 테스트 데이터 로드
# ============================================
print("\n[2단계] 테스트 데이터 로드")

with open(DATA_DIR / "test.json", 'r', encoding='utf-8') as f:
    test_data = json.load(f)

print(f"테스트 샘플 수: {len(test_data)}")

# 클래스별 개수
label_counts = {}
for item in test_data:
    label = item['label']
    label_counts[label] = label_counts.get(label, 0) + 1

for label, count in label_counts.items():
    print(f"  - {label}: {count}개")

# ============================================
# 3. 테스트 평가
# ============================================
print("\n[3단계] 테스트 세트 평가")

predictions = []
true_labels = []

print("테스트 데이터 추론 중...")

for i, item in enumerate(test_data):
    # 이미지 로드
    image_path = DATA_DIR / item['image']
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        continue

    # 실제 라벨
    true_label = LABEL_MAP[item['label']]

    # 프롬프트 (boi_token 포함)
    boi_token = processor.tokenizer.boi_token
    user_message = item['conversations'][0]['content']
    prompt = f"{boi_token}{user_message}"

    # 인코딩
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    )

    # token_type_ids가 없으면 생성
    if 'token_type_ids' not in inputs:
        inputs['token_type_ids'] = torch.zeros_like(inputs['input_ids'])

    # GPU로 이동
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id
        )

    # 디코딩
    generated_text = processor.decode(outputs[0], skip_special_tokens=True)

    # 라벨 추출 (휴리스틱)
    generated_lower = generated_text.lower()
    if "malignant" in generated_lower or "악성" in generated_lower:
        pred_label = 2
    elif "benign" in generated_lower or "양성" in generated_lower:
        pred_label = 1
    else:
        pred_label = 0  # normal

    predictions.append(pred_label)
    true_labels.append(true_label)

    # 진행 상황 출력
    if (i + 1) % 20 == 0:
        print(f"  진행: {i+1}/{len(test_data)}")

    # 처음 5개 샘플 결과 출력
    if i < 5:
        print(f"\n샘플 {i+1}:")
        print(f"  실제: {LABEL_NAMES[true_label]}")
        print(f"  예측: {LABEL_NAMES[pred_label]}")
        print(f"  생성된 텍스트: {generated_text[:200]}...")

# ============================================
# 4. 평가 지표 계산
# ============================================
print("\n[4단계] 평가 지표 계산")

accuracy = accuracy_score(true_labels, predictions)
print(f"\n테스트 정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n분류 리포트:")
print(classification_report(true_labels, predictions, target_names=LABEL_NAMES))

# Confusion Matrix
cm = confusion_matrix(true_labels, predictions)
print("\nConfusion Matrix:")
print(cm)

# ============================================
# 5. 결과 시각화 및 저장
# ============================================
print("\n[5단계] 결과 저장")

# Confusion Matrix 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=LABEL_NAMES,
            yticklabels=LABEL_NAMES)
plt.title('Confusion Matrix - MedGemma CT-LoRA')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"Confusion Matrix 저장: {OUTPUT_DIR / 'confusion_matrix.png'}")

# 결과 JSON 저장
results = {
    "model": MODEL_ID,
    "adapter_path": str(ADAPTER_PATH),
    "method": "CT-LoRA Fine-tuning",
    "test_accuracy": float(accuracy),
    "test_samples": len(test_data),
    "confusion_matrix": cm.tolist(),
    "classification_report": classification_report(true_labels, predictions, target_names=LABEL_NAMES, output_dict=True)
}

with open(OUTPUT_DIR / 'test_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"결과 저장: {OUTPUT_DIR / 'test_results.json'}")

# 클래스별 정확도 계산
print("\n클래스별 성능:")
for i, label_name in enumerate(LABEL_NAMES):
    mask = [tl == i for tl in true_labels]
    if sum(mask) > 0:
        class_predictions = [p for p, m in zip(predictions, mask) if m]
        class_true = [t for t, m in zip(true_labels, mask) if m]
        class_acc = accuracy_score(class_true, class_predictions)
        print(f"  {label_name}: {class_acc:.4f} ({class_acc*100:.2f}%)")

print("\n" + "=" * 60)
print("테스트 평가 완료!")
print("=" * 60)
print(f"전체 정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"결과 디렉토리: {OUTPUT_DIR}")
