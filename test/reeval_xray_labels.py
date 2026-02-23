"""
X-ray LoRA 라벨 추출 재평가 스크립트

모델 재추론 없이 저장된 generated_text만으로
label extraction 로직을 바꿔가며 정확도를 빠르게 비교.

사용법:
    python reeval_xray_labels.py

입력:
    xray_lora_results/all_generated_texts.jsonl  (test_medgemma_lora_xray.py 실행 후 생성)

출력:
    xray_lora_results/reeval_results.json
"""

import json
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = Path("/home/students/cs/doctor/AI_Doctor/test/xray_lora_results")
INPUT_FILE = OUTPUT_DIR / "all_generated_texts.jsonl"

LABEL_MAP = {"normal": 0, "abnormal": 1}
LABEL_NAMES = ["normal", "abnormal"]

# ============================================
# 재평가할 label extraction 함수들 정의
# ============================================

def extract_v1_original(text):
    """원본 로직 (버그 있음: 프롬프트 오염 + '정상 폐' 누락)"""
    tl = text.lower().strip()
    fp = tl[:100]
    if any(p in fp for p in ['비정상입니다', '비정상 소견', '비정상 x-ray']):
        return 1
    if any(p in fp for p in ['정상입니다', '정상 소견', '정상 x-ray', '정상 범위']):
        return 0
    scores = {'normal': 0, 'abnormal': 0}
    if '정상' in tl: scores['normal'] += 10
    if '이상 없' in tl or '소견 없' in tl: scores['normal'] += 10
    if '관찰되지 않' in tl: scores['normal'] += 8
    if '비정상' in tl: scores['abnormal'] += 15
    if '이상 소견이' in tl or '이상소견이' in tl: scores['abnormal'] += 12
    if '이상 소견' in tl and '없' not in tl: scores['abnormal'] += 10
    if '병변' in tl:
        if '병변이 없' in tl or '병변 없' in tl: scores['normal'] += 5
        else: scores['abnormal'] += 8
    if '추가 검사' in tl or '전문의' in tl: scores['abnormal'] += 6
    return 1 if scores['abnormal'] > scores['normal'] else 0


def extract_v2_fixed(text):
    """수정된 로직 (프롬프트 제거 + 키워드 보완)"""
    tl = text.lower().strip()
    # 프롬프트 이후 응답만 추출
    rt = tl
    for ending in ['판단해주세요.', '평가해주세요.', '알려주세요.', '분석해주세요.']:
        idx = tl.find(ending)
        if idx != -1:
            rt = tl[idx + len(ending):].strip()
            break

    if any(p in rt for p in ['비정상입니다', '비정상 소견', '비정상 x-ray']):
        return 1
    if any(p in rt for p in ['정상입니다', '정상 소견', '정상 x-ray', '정상 폐', '정상 범위']):
        return 0

    scores = {'normal': 0, 'abnormal': 0}
    if '정상' in rt: scores['normal'] += 10
    if '이상 없' in rt or '소견 없' in rt: scores['normal'] += 10
    if '관찰되지 않' in rt or '보이지 않' in rt: scores['normal'] += 8
    if '병적 소견' in rt and ('없' in rt or '관찰되지' in rt): scores['normal'] += 6
    if '비정상' in rt: scores['abnormal'] += 15
    if '이상 소견이' in rt or '이상소견이' in rt: scores['abnormal'] += 12
    if '이상 소견' in rt and '없' not in rt and '관찰되지' not in rt: scores['abnormal'] += 10
    if '병변' in rt:
        if '병변이 없' in rt or '병변 없' in rt: scores['normal'] += 5
        else: scores['abnormal'] += 8
    if '추가 검사' in rt or '전문의' in rt: scores['abnormal'] += 6
    if '추가 진료' in rt or '추가 검진' in rt: scores['abnormal'] += 4
    return 1 if scores['abnormal'] > scores['normal'] else 0


# 여기에 새 버전 추가 가능
# def extract_v3_xxx(text): ...

EXTRACTORS = {
    "v1_original": extract_v1_original,
    "v2_fixed":    extract_v2_fixed,
}

# ============================================
# 데이터 로드
# ============================================
print("=" * 60)
print("X-ray Label Extraction 재평가")
print("=" * 60)

if not INPUT_FILE.exists():
    print(f"\n[오류] 입력 파일 없음: {INPUT_FILE}")
    print("먼저 test_medgemma_lora_xray.py 를 실행해 generated_text를 저장하세요.")
    exit(1)

records = []
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"\n로드된 샘플: {len(records)}개")

true_labels = [LABEL_MAP[r['true_label']] for r in records]
texts       = [r['generated_text'] for r in records]

# 프롬프트 종류별 분포 출력
prompt_counts = {}
for r in records:
    tl = r['generated_text'].strip().lower()
    for ending in ['판단해주세요.', '평가해주세요.', '알려주세요.', '분석해주세요.']:
        idx = tl.find(ending)
        if idx != -1:
            prompt = tl[:idx + len(ending)].strip()
            prompt_counts[prompt] = prompt_counts.get(prompt, 0) + 1
            break

print("\n[프롬프트 분포]")
for p, cnt in sorted(prompt_counts.items(), key=lambda x: -x[1]):
    contaminated = '이상 소견이' in p
    print(f"  {'[오염]' if contaminated else '[정상]'} {cnt:5d}개  {p}")

# ============================================
# 각 버전 평가
# ============================================
print("\n" + "=" * 60)
all_results = {}

for name, fn in EXTRACTORS.items():
    preds = [fn(t) for t in texts]
    acc   = accuracy_score(true_labels, preds)
    cm    = confusion_matrix(true_labels, preds)
    report = classification_report(
        true_labels, preds, target_names=LABEL_NAMES,
        output_dict=True, zero_division=0
    )

    print(f"\n[{name}]")
    print(f"  정확도: {acc*100:.2f}%")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0][0]:5d} FP={cm[0][1]:5d}  (normal 샘플 {cm[0][0]+cm[0][1]}개)")
    print(f"    FN={cm[1][0]:5d} TP={cm[1][1]:5d}  (abnormal 샘플 {cm[1][0]+cm[1][1]}개)")
    print(f"  Normal   precision={report['normal']['precision']:.4f}  recall={report['normal']['recall']:.4f}  f1={report['normal']['f1-score']:.4f}")
    print(f"  Abnormal precision={report['abnormal']['precision']:.4f}  recall={report['abnormal']['recall']:.4f}  f1={report['abnormal']['f1-score']:.4f}")

    all_results[name] = {
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

# ============================================
# 버전 간 변경 분석
# ============================================
versions = list(EXTRACTORS.keys())
if len(versions) >= 2:
    v1_name, v2_name = versions[0], versions[1]
    v1_fn = EXTRACTORS[v1_name]
    v2_fn = EXTRACTORS[v2_name]

    v1_preds = [v1_fn(t) for t in texts]
    v2_preds = [v2_fn(t) for t in texts]

    changed_to_correct   = 0
    changed_to_incorrect = 0
    change_details = []

    for i, (true, p1, p2) in enumerate(zip(true_labels, v1_preds, v2_preds)):
        if p1 != p2:
            if p2 == true:
                changed_to_correct += 1
            else:
                changed_to_incorrect += 1
            change_details.append({
                "idx": i,
                "true": LABEL_NAMES[true],
                "v1_pred": LABEL_NAMES[p1],
                "v2_pred": LABEL_NAMES[p2],
                "text_snippet": records[i]['generated_text'].strip()[-80:]
            })

    total_changed = changed_to_correct + changed_to_incorrect
    print(f"\n[{v1_name} → {v2_name} 변경 분석]")
    print(f"  총 변경: {total_changed}개")
    print(f"  오답→정답: {changed_to_correct}개  ✅")
    print(f"  정답→오답: {changed_to_incorrect}개  ❌")

    if change_details:
        print(f"\n  변경 샘플 예시 (최대 10개):")
        for d in change_details[:10]:
            sign = '✅' if d['v2_pred'] == d['true'] else '❌'
            print(f"  {sign} true={d['true']:8s} {d['v1_pred']}→{d['v2_pred']} | ...{d['text_snippet']}")

# ============================================
# 결과 저장
# ============================================
save_path = OUTPUT_DIR / 'reeval_results.json'
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n결과 저장: {save_path}")
print("\n" + "=" * 60)
print("재평가 완료!")
print("=" * 60)
