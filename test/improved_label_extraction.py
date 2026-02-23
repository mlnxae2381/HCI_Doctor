"""
개선된 라벨 추출 함수
더 명확하고 간단한 로직으로 정상/비정상 판단
"""

def extract_label_improved(text):
    """
    개선된 X-ray 응답 라벨 추출
    - 첫 문장 우선 분석
    - 명확한 키워드 중심
    - 부정 표현 처리
    """
    text_lower = text.lower().strip()

    # 첫 100자에서 판단 (첫 문장 우선)
    first_part = text_lower[:100]

    # 1. 최우선 키워드 체크 (첫 부분에서)
    # 비정상 명시
    if any(phrase in first_part for phrase in ['비정상입니다', '비정상 소견', '비정상 x-ray']):
        return 1  # Abnormal

    # 정상 명시
    if any(phrase in first_part for phrase in ['정상입니다', '정상 소견', '정상 x-ray', '정상 범위']):
        return 0  # Normal

    # 2. 전체 텍스트에서 점수 계산
    scores = {'normal': 0, 'abnormal': 0}

    # === Normal 시그널 ===
    # "정상" 키워드
    if '정상' in text_lower:
        scores['normal'] += 10

    # 부정 표현 (이상 없음, 소견 없음 등)
    if '이상 없' in text_lower or '소견 없' in text_lower:
        scores['normal'] += 10

    if '관찰되지 않' in text_lower:
        scores['normal'] += 8

    # === Abnormal 시그널 ===
    # "비정상" 키워드
    if '비정상' in text_lower:
        scores['abnormal'] += 15

    # "이상 소견이" (긍정형만)
    if '이상 소견이' in text_lower or '이상소견이' in text_lower:
        scores['abnormal'] += 12

    # "이상 소견" (일반)
    if '이상 소견' in text_lower and '없' not in text_lower:
        scores['abnormal'] += 10

    # "병변"
    if '병변' in text_lower:
        # 부정형 체크
        if '병변이 없' in text_lower or '병변 없' in text_lower:
            scores['normal'] += 5
        else:
            scores['abnormal'] += 8

    # "추가 검사", "전문의"
    if '추가 검사' in text_lower or '전문의' in text_lower:
        scores['abnormal'] += 6

    # 3. 점수 기반 판단
    if scores['abnormal'] > scores['normal']:
        return 1
    elif scores['normal'] > scores['abnormal']:
        return 0
    else:
        # 동점이면 기본값 normal (보수적 판단)
        return 0

# 테스트
if __name__ == "__main__":
    test_cases = [
        ("정상입니다. 이상 소견 없습니다.", 0),
        ("정상 흉부 X-ray입니다.", 0),
        ("비정상입니다. 이상 소견이 관찰됩니다.", 1),
        ("비정상 소견입니다. 추가 검사가 필요합니다.", 1),
        ("정상 폐 X-ray입니다. 병적 소견이 관찰되지 않습니다.", 0),
        ("비정상 소견으로 판단됩니다. 폐 병변이 의심되어 전문의 상담이 필요합니다.", 1),
    ]

    print("라벨 추출 함수 테스트:")
    print("=" * 60)

    for text, expected in test_cases:
        result = extract_label_improved(text)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{text[:40]}...'")
        print(f"   예상: {expected}, 결과: {result}")
        print()
