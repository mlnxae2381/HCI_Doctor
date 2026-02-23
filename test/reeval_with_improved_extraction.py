"""
기존 테스트 결과에 개선된 라벨 추출 함수 적용
재평가하여 성능 개선 확인
"""

import json
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 경로 설정
RESULTS_FILE = Path("/home/students/cs/doctor/AI_Doctor/test/xray_lora_results/test_results_xray.json")
LABEL_MAP = {"normal": 0, "abnormal": 1}
LABEL_NAMES = ["normal", "abnormal"]

def extract_label_improved(text):
    """개선된 라벨 추출 함수"""
    text_lower = text.lower().strip()
    first_part = text_lower[:100]

    # 최우선 키워드 체크
    if any(phrase in first_part for phrase in ['비정상입니다', '비정상 소견', '비정상 x-ray']):
        return 1

    if any(phrase in first_part for phrase in ['정상입니다', '정상 소견', '정상 x-ray', '정상 범위']):
        return 0

    # 점수 계산
    scores = {'normal': 0, 'abnormal': 0}

    if '정상' in text_lower:
        scores['normal'] += 10

    if '이상 없' in text_lower or '소견 없' in text_lower:
        scores['normal'] += 10

    if '관찰되지 않' in text_lower:
        scores['normal'] += 8

    if '비정상' in text_lower:
        scores['abnormal'] += 15

    if '이상 소견이' in text_lower or '이상소견이' in text_lower:
        scores['abnormal'] += 12

    if '이상 소견' in text_lower and '없' not in text_lower:
        scores['abnormal'] += 10

    if '병변' in text_lower:
        if '병변이 없' in text_lower or '병변 없' in text_lower:
            scores['normal'] += 5
        else:
            scores['abnormal'] += 8

    if '추가 검사' in text_lower or '전문의' in text_lower:
        scores['abnormal'] += 6

    if scores['abnormal'] > scores['normal']:
        return 1
    elif scores['normal'] > scores['abnormal']:
        return 0
    else:
        return 0

print("=" * 60)
print("개선된 라벨 추출 로직으로 재평가")
print("=" * 60)

# 기존 결과 로드
with open(RESULTS_FILE, 'r') as f:
    results = json.load(f)

print(f"\n원본 정확도: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
print(f"테스트 샘플 수: {results['test_samples']}")

# 상세 결과에서 재평가
detailed = results['detailed_results']

true_labels = []
old_predictions = []
new_predictions = []

for item in detailed:
    true_label = LABEL_MAP[item['true_label']]
    old_pred = LABEL_MAP[item['pred_label']]

    # 개선된 함수로 재추출
    new_pred = extract_label_improved(item['generated_text'])

    true_labels.append(true_label)
    old_predictions.append(old_pred)
    new_predictions.append(new_pred)

# 샘플 비교 (처음 20개)
print("\n" + "=" * 60)
print("샘플 비교 (처음 20개)")
print("=" * 60)

for i in range(min(20, len(detailed))):
    item = detailed[i]
    true_label = item['true_label']
    old_pred = item['pred_label']
    new_pred = LABEL_NAMES[new_predictions[i]]

    old_correct = "✅" if old_pred == true_label else "❌"
    new_correct = "✅" if new_pred == true_label else "❌"

    if old_pred != new_pred:
        print(f"\n샘플 {i}:")
        print(f"  실제: {true_label}")
        print(f"  이전: {old_pred} {old_correct} → 개선: {new_pred} {new_correct}")
        print(f"  텍스트: {item['generated_text'][:80]}...")

# 전체 통계
print("\n" + "=" * 60)
print("전체 통계 비교")
print("=" * 60)

# 참고: detailed_results는 처음 20개만 있으므로, 실제 전체 재평가는 별도 필요
print(f"\n📊 샘플 20개 기준:")
print(f"  이전 정확도: {accuracy_score(true_labels[:20], old_predictions[:20]):.4f}")
print(f"  개선 정확도: {accuracy_score(true_labels[:20], new_predictions[:20]):.4f}")

print("\n" + "=" * 60)
print("✅ 재평가 완료!")
print("=" * 60)
print("\n💡 전체 11,213개 샘플을 재평가하려면:")
print("   python test_medgemma_lora_xray.py")
print("=" * 60)
