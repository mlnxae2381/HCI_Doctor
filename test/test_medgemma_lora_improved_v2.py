"""
MedGemma CT-LoRA 테스트 스크립트 (개선된 라벨 추출 v2)
학습된 LoRA 어댑터를 불러와서 테스트 세트 평가
- 프롬프트 영향 제거: 모델 응답만 점수 계산
- 개선된 점수 체계: 더 정교한 패턴 매칭
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU 1번 사용 (0번은 메모리 부족)

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
BASE_DIR = Path("/home/students/cs/doctor/AI_Doctor/data")
DATA_DIR = BASE_DIR / "processed"
OUTPUT_DIR = Path("/home/students/cs/doctor/AI_Doctor/test/ct_lora_results")
ADAPTER_PATH = OUTPUT_DIR / "ct_lora_adapter"

MODEL_ID = "google/medgemma-1.5-4b-it"

# 클래스 매핑
LABEL_MAP = {"normal": 0, "benign": 1, "malignant": 2}
LABEL_NAMES = ["normal", "benign", "malignant"]

print("=" * 60)
print("MedGemma CT-LoRA 테스트 평가 (개선된 버전 v2)")
print("=" * 60)


# ============================================
# 프롬프트 제거 함수
# ============================================
def remove_prompt_from_text(full_text, prompt):
    """
    생성된 텍스트에서 프롬프트 부분을 제거하고 모델의 실제 응답만 추출
    """
    # 프롬프트가 텍스트에 포함되어 있으면 제거
    if prompt in full_text:
        response_only = full_text.replace(prompt, "").strip()
    else:
        # 프롬프트가 정확히 일치하지 않으면 텍스트 전체를 응답으로 간주
        response_only = full_text.strip()

    return response_only


# ============================================
# 개선된 라벨 추출 함수 v2
# ============================================
def extract_label_smart_v2(text):
    """
    개선된 스마트 라벨 추출: 맥락을 고려한 점수 시스템 v2
    - 더 정교한 패턴 매칭
    - 부정 표현 처리 강화
    - 문맥 기반 점수 조정
    """
    text_lower = text.lower()

    # 점수 시스템 (높을수록 해당 클래스 가능성 높음)
    scores = {'normal': 0, 'benign': 0, 'malignant': 0}

    # ===== NORMAL 시그널 =====
    # 1. 명확한 정상 표현
    if re.search(r'정상\s*(소견|폐|ct)', text_lower):
        scores['normal'] += 6
    elif '정상입니다' in text_lower or '정상이다' in text_lower:
        scores['normal'] += 5
    elif '정상' in text_lower:
        scores['normal'] += 3

    # 2. 이상 없음
    if re.search(r'이상\s*소견.*없', text_lower):
        scores['normal'] += 5
    if re.search(r'병변.*관찰되지\s*않', text_lower):
        scores['normal'] += 5
    if re.search(r'(종양|결절).*없', text_lower):
        scores['normal'] += 4

    # 3. 정상 구조 언급
    if '정상적으로 보입니다' in text_lower:
        scores['normal'] += 3

    # ===== BENIGN 시그널 =====
    # 1. 명확한 양성 표현 (가중치 높임)
    if re.search(r'양성\s*(종양|병변|결절)', text_lower):
        scores['benign'] += 8
    if '양성으로 판단' in text_lower or '양성으로 평가' in text_lower:
        scores['benign'] += 7
    if '양성 소견' in text_lower:
        scores['benign'] += 6
    if re.search(r'양성.*의심', text_lower):
        scores['benign'] += 6

    # 2. 악성 부정 표현 (양성의 강한 시그널)
    if re.search(r'악성\s*가능성.*낮', text_lower):
        scores['benign'] += 6
    if re.search(r'악성.*특징.*보이지\s*않', text_lower):
        scores['benign'] += 5
    if re.search(r'악성.*특징.*없', text_lower):
        scores['benign'] += 5
    if '악성이 아니' in text_lower or '악성은 아니' in text_lower:
        scores['benign'] += 5
    if re.search(r'악성.*배제', text_lower):
        scores['benign'] += 4

    # 3. 양성 특징
    if '경계가 명확' in text_lower or '경계.*명확' in text_lower:
        scores['benign'] += 3
    if '석회화' in text_lower:
        scores['benign'] += 2
    if '경과 관찰' in text_lower or '추적 관찰' in text_lower:
        scores['benign'] += 2

    # ===== MALIGNANT 시그널 =====
    # 1. 명확한 악성 표현
    malignant_negation_patterns = [
        r'악성\s*가능성.*낮',
        r'악성.*특징.*없',
        r'악성.*특징.*보이지\s*않',
        r'악성.*아니',
        r'악성.*배제',
    ]

    has_malignant_negation = any(re.search(pattern, text_lower) for pattern in malignant_negation_patterns)

    if '악성' in text_lower and not has_malignant_negation:
        # 악성이 단독으로 나오거나 긍정적 맥락에서 나오면
        if re.search(r'악성\s*(병변|종양|소견)', text_lower):
            scores['malignant'] += 8
        elif re.search(r'악성\s*가능성.*높', text_lower):
            scores['malignant'] += 7
        elif re.search(r'악성.*의심', text_lower) and '낮' not in text_lower:
            scores['malignant'] += 6
        else:
            scores['malignant'] += 5

    # 2. 폐암 언급
    if '폐암' in text_lower:
        if re.search(r'폐암.*아니', text_lower) or re.search(r'폐암.*없', text_lower):
            scores['normal'] += 3  # 폐암 부정은 정상 시그널
        else:
            if '폐암 의심' in text_lower:
                scores['malignant'] += 7
            else:
                scores['malignant'] += 6

    # 3. 악성 관련 표현
    if '악성 병변' in text_lower:
        scores['malignant'] += 6

    # 4. 악성 특징
    if '불규칙한 경계' in text_lower or '경계 불규칙' in text_lower:
        scores['malignant'] += 4
    if re.search(r'(주변|조직).*침윤', text_lower):
        scores['malignant'] += 4
    if '림프절 전이' in text_lower or '전이' in text_lower:
        scores['malignant'] += 4

    # 5. 추가 검사 필요 (악성 의심 시그널)
    if 'pet-ct' in text_lower or 'pet ct' in text_lower:
        scores['malignant'] += 2
    if '조직 검사' in text_lower or '생검' in text_lower:
        scores['malignant'] += 2
    if '확진' in text_lower:
        scores['malignant'] += 1

    # 가장 높은 점수의 라벨 반환
    max_label = max(scores, key=scores.get)
    max_score = scores[max_label]

    # 점수가 0이면 기본값 normal
    if max_score == 0:
        return 0, scores

    return LABEL_MAP[max_label], scores


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
# 3. 테스트 평가 (프롬프트 제거 적용)
# ============================================
print("\n[3단계] 테스트 세트 평가 (프롬프트 제거 + 개선된 라벨 추출)")

predictions = []
true_labels = []
detailed_results = []  # 상세 결과 저장

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

    # **핵심 개선**: 프롬프트 제거 후 응답만 점수 계산
    response_only = remove_prompt_from_text(generated_text, user_message)

    # 개선된 라벨 추출 (응답 부분만 사용)
    pred_label, scores = extract_label_smart_v2(response_only)

    predictions.append(pred_label)
    true_labels.append(true_label)

    # 상세 결과 저장
    detailed_results.append({
        'sample_id': i,
        'true_label': LABEL_NAMES[true_label],
        'pred_label': LABEL_NAMES[pred_label],
        'scores': scores,
        'prompt': user_message,
        'response_only': response_only,
        'generated_text': generated_text,
        'correct': true_label == pred_label
    })

    # 진행 상황 출력
    if (i + 1) % 20 == 0:
        print(f"  진행: {i+1}/{len(test_data)}")

    # 처음 10개 샘플 결과 출력 (상세)
    if i < 10:
        print(f"\n샘플 {i+1}:")
        print(f"  실제: {LABEL_NAMES[true_label]}")
        print(f"  예측: {LABEL_NAMES[pred_label]} {'✅' if true_label == pred_label else '❌'}")
        print(f"  점수: normal={scores['normal']}, benign={scores['benign']}, malignant={scores['malignant']}")
        print(f"  프롬프트: {user_message}")
        print(f"  모델 응답: {response_only[:100]}...")

# ============================================
# 4. 평가 지표 계산
# ============================================
print("\n[4단계] 평가 지표 계산")

accuracy = accuracy_score(true_labels, predictions)
print(f"\n✨ 테스트 정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n분류 리포트:")
print(classification_report(true_labels, predictions, target_names=LABEL_NAMES, zero_division=0))

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
plt.title('Confusion Matrix - MedGemma CT-LoRA (v2)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confusion_matrix_v2.png', dpi=300, bbox_inches='tight')
print(f"Confusion Matrix 저장: {OUTPUT_DIR / 'confusion_matrix_v2.png'}")

# 결과 JSON 저장
results = {
    "model": MODEL_ID,
    "adapter_path": str(ADAPTER_PATH),
    "method": "CT-LoRA Fine-tuning (Improved Label Extraction v2 - Prompt Removed)",
    "test_accuracy": float(accuracy),
    "test_samples": len(test_data),
    "confusion_matrix": cm.tolist(),
    "classification_report": classification_report(true_labels, predictions, target_names=LABEL_NAMES, output_dict=True, zero_division=0),
    "detailed_results": detailed_results[:20]  # 처음 20개만 저장 (용량 고려)
}

with open(OUTPUT_DIR / 'test_results_v2.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"결과 저장: {OUTPUT_DIR / 'test_results_v2.json'}")

# 클래스별 정확도 계산
print("\n클래스별 성능:")
for i, label_name in enumerate(LABEL_NAMES):
    mask = [tl == i for tl in true_labels]
    if sum(mask) > 0:
        class_predictions = [p for p, m in zip(predictions, mask) if m]
        class_true = [t for t, m in zip(true_labels, mask) if m]
        class_acc = accuracy_score(class_true, class_predictions)
        print(f"  {label_name}: {class_acc:.4f} ({class_acc*100:.2f}%) - {sum(mask)}개 샘플")

# 잘못 분류된 샘플 분석
print("\n잘못 분류된 샘플 분석:")
misclassified = [r for r in detailed_results if not r['correct']]
print(f"총 {len(misclassified)}개 샘플 잘못 분류")

# 각 클래스별 오분류 패턴
for true_label_name in LABEL_NAMES:
    misclass_for_label = [r for r in misclassified if r['true_label'] == true_label_name]
    if misclass_for_label:
        print(f"\n{true_label_name} 오분류 ({len(misclass_for_label)}개):")
        pred_distribution = {}
        for r in misclass_for_label:
            pred = r['pred_label']
            pred_distribution[pred] = pred_distribution.get(pred, 0) + 1
        for pred, count in pred_distribution.items():
            print(f"  → {pred}로 분류: {count}개")

print("\n" + "=" * 60)
print("테스트 평가 완료!")
print("=" * 60)
print(f"✨ 전체 정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"📁 결과 디렉토리: {OUTPUT_DIR}")
print("\n개선 사항 v2:")
print("  - 프롬프트 제거: 모델 응답만 점수 계산")
print("  - 더 정교한 점수 체계")
print("  - 부정 표현 처리 강화")
print("  - 문맥 기반 점수 조정")
