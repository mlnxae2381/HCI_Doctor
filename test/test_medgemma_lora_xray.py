"""
MedGemma X-ray LoRA 테스트 스크립트
학습된 LoRA 어댑터로 X-ray 테스트 세트 평가
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU 1번 사용

import json
import re
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
BASE_DIR = Path("/home/students/cs/doctor/AI_Doctor/data/AI_Doctor_Data/Chest_Xray/final_dataset")
DATA_DIR = Path("/home/students/cs/doctor/AI_Doctor/data/xray_processed")
OUTPUT_DIR = Path("/home/students/cs/doctor/AI_Doctor/test/xray_lora_results")
ADAPTER_PATH = OUTPUT_DIR / "xray_lora_adapter"

MODEL_ID = "google/medgemma-1.5-4b-it"

# 클래스 매핑
LABEL_MAP = {"normal": 0, "abnormal": 1}
LABEL_NAMES = ["normal", "abnormal"]

print("=" * 60)
print("MedGemma X-ray LoRA 테스트 평가")
print("=" * 60)

# ============================================
# 라벨 추출 함수
# ============================================
def extract_label_smart(text):
    """
    개선된 X-ray 응답 라벨 추출
    - 프롬프트 제거 후 응답 부분만 분석 (프롬프트 오염 방지)
    - 명확한 키워드 중심
    """
    text_lower = text.lower().strip()

    # 프롬프트 이후 생성된 응답 부분만 추출
    # (프롬프트의 "이상 소견이 있는지" 같은 표현이 scoring을 오염시키는 것 방지)
    response_text = text_lower
    for ending in ['판단해주세요.', '평가해주세요.', '알려주세요.', '분석해주세요.']:
        idx = text_lower.find(ending)
        if idx != -1:
            response_text = text_lower[idx + len(ending):].strip()
            break

    # 1. 직접 키워드 체크 (명확한 판단)
    if any(phrase in response_text for phrase in ['비정상입니다', '비정상 소견', '비정상 x-ray']):
        return 1, {'normal': 0, 'abnormal': 15}

    if any(phrase in response_text for phrase in ['정상입니다', '정상 소견', '정상 x-ray', '정상 폐', '정상 범위']):
        return 0, {'normal': 10, 'abnormal': 0}

    # 2. 점수 계산 (응답 부분만)
    scores = {'normal': 0, 'abnormal': 0}

    if '정상' in response_text:
        scores['normal'] += 10

    if '이상 없' in response_text or '소견 없' in response_text:
        scores['normal'] += 10

    if '관찰되지 않' in response_text or '보이지 않' in response_text:
        scores['normal'] += 8

    if '병적 소견' in response_text and ('없' in response_text or '관찰되지' in response_text):
        scores['normal'] += 6

    if '비정상' in response_text:
        scores['abnormal'] += 15

    if '이상 소견이' in response_text or '이상소견이' in response_text:
        scores['abnormal'] += 12

    if '이상 소견' in response_text and '없' not in response_text and '관찰되지' not in response_text:
        scores['abnormal'] += 10

    if '병변' in response_text:
        if '병변이 없' in response_text or '병변 없' in response_text:
            scores['normal'] += 5
        else:
            scores['abnormal'] += 8

    if '추가 검사' in response_text or '전문의' in response_text:
        scores['abnormal'] += 6

    if '추가 진료' in response_text or '추가 검진' in response_text:
        scores['abnormal'] += 4

    # 3. 점수 기반 판단
    if scores['abnormal'] > scores['normal']:
        return 1, scores
    return 0, scores  # 동점이면 normal

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
detailed_results = []

print("테스트 데이터 추론 중...")

for i, item in enumerate(test_data):
    # 이미지 로드
    image_path = BASE_DIR / item['image']
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        continue

    # 실제 라벨
    true_label = LABEL_MAP[item['label']]

    # 프롬프트
    boi_token = processor.tokenizer.boi_token
    user_message = item['conversations'][0]['content']
    prompt = f"{boi_token}{user_message}"

    # 인코딩
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    )

    if 'token_type_ids' not in inputs:
        inputs['token_type_ids'] = torch.zeros_like(inputs['input_ids'])

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

    # 라벨 추출
    pred_label, scores = extract_label_smart(generated_text)

    predictions.append(pred_label)
    true_labels.append(true_label)

    detailed_results.append({
        'sample_id': i,
        'true_label': LABEL_NAMES[true_label],
        'pred_label': LABEL_NAMES[pred_label],
        'scores': scores,
        'generated_text': generated_text,
        'correct': true_label == pred_label
    })

    # 진행 상황
    if (i + 1) % 100 == 0:
        print(f"  진행: {i+1}/{len(test_data)}")

    # 처음 10개 출력
    if i < 10:
        print(f"\n샘플 {i+1}:")
        print(f"  실제: {LABEL_NAMES[true_label]}")
        print(f"  예측: {LABEL_NAMES[pred_label]} {'✅' if true_label == pred_label else '❌'}")
        print(f"  점수: normal={scores['normal']}, abnormal={scores['abnormal']}")
        print(f"  생성: {generated_text[:100]}...")

# ============================================
# 4. 평가 지표 계산
# ============================================
print("\n[4단계] 평가 지표 계산")

accuracy = accuracy_score(true_labels, predictions)
print(f"\n✨ 테스트 정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n분류 리포트:")
print(classification_report(true_labels, predictions, target_names=LABEL_NAMES, zero_division=0))

cm = confusion_matrix(true_labels, predictions)
print("\nConfusion Matrix:")
print(cm)

# ============================================
# 5. 결과 저장
# ============================================
print("\n[5단계] 결과 저장")

# Confusion Matrix 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=LABEL_NAMES,
            yticklabels=LABEL_NAMES)
plt.title('Confusion Matrix - MedGemma X-ray LoRA')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confusion_matrix_xray.png', dpi=300, bbox_inches='tight')
print(f"Confusion Matrix 저장: {OUTPUT_DIR / 'confusion_matrix_xray.png'}")

# 결과 JSON 저장
results = {
    "model": MODEL_ID,
    "adapter_path": str(ADAPTER_PATH),
    "method": "X-ray LoRA Fine-tuning",
    "test_accuracy": float(accuracy),
    "test_samples": len(test_data),
    "confusion_matrix": cm.tolist(),
    "classification_report": classification_report(true_labels, predictions, target_names=LABEL_NAMES, output_dict=True, zero_division=0),
    "detailed_results": detailed_results[:20]
}

with open(OUTPUT_DIR / 'test_results_xray.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"결과 저장: {OUTPUT_DIR / 'test_results_xray.json'}")

# 전체 generated_text 별도 저장 (label extraction 재평가용)
all_texts_path = OUTPUT_DIR / 'all_generated_texts.jsonl'
with open(all_texts_path, 'w', encoding='utf-8') as f:
    for r in detailed_results:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print(f"전체 생성 텍스트 저장: {all_texts_path} ({len(detailed_results)}개)")

# 클래스별 정확도
print("\n클래스별 성능:")
for i, label_name in enumerate(LABEL_NAMES):
    mask = [tl == i for tl in true_labels]
    if sum(mask) > 0:
        class_predictions = [p for p, m in zip(predictions, mask) if m]
        class_true = [t for t, m in zip(true_labels, mask) if m]
        class_acc = accuracy_score(class_true, class_predictions)
        print(f"  {label_name}: {class_acc:.4f} ({class_acc*100:.2f}%) - {sum(mask)}개 샘플")

# 오분류 분석
print("\n잘못 분류된 샘플:")
misclassified = [r for r in detailed_results if not r['correct']]
print(f"총 {len(misclassified)}개")

print("\n" + "=" * 60)
print("테스트 평가 완료!")
print("=" * 60)
print(f"✨ 전체 정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"📁 결과 디렉토리: {OUTPUT_DIR}")
