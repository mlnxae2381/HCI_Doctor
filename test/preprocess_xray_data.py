"""
X-ray 데이터를 MedGemma 학습 형식으로 전처리
CT 데이터와 동일한 JSON 형식 생성
"""

import json
from pathlib import Path
import random
from tqdm import tqdm

# 경로 설정
DATA_DIR = Path("/home/students/cs/doctor/AI_Doctor/data/AI_Doctor_Data/Chest_Xray/final_dataset")
OUTPUT_DIR = Path("/home/students/cs/doctor/AI_Doctor/data/xray_processed")
OUTPUT_DIR.mkdir(exist_ok=True)

# 클래스 매핑
LABEL_MAP = {"Normal": "normal", "Abnormal": "abnormal"}

# 프롬프트 템플릿 (다양성 확보)
USER_PROMPTS = [
    "이 흉부 X-ray 이미지를 분석하여 이상 소견이 있는지 판단해주세요.",
    "이 X-ray 영상에서 폐 병변이 관찰되는지 평가해주세요.",
    "흉부 X-ray를 보고 정상/비정상 여부를 판단해주세요.",
    "이 X-ray 이미지에서 특이 소견이 있는지 분석해주세요.",
    "폐 X-ray 영상을 평가하여 이상 유무를 알려주세요.",
]

ASSISTANT_RESPONSES = {
    "normal": [
        "정상입니다. 이상 소견 없습니다.",
        "정상 흉부 X-ray입니다.",
        "정상 소견입니다.",
        "정상 범위입니다.",
        "정상입니다.",
    ],
    "abnormal": [
        "비정상입니다. 이상 소견이 관찰됩니다.",
        "비정상 소견입니다. 추가 검사가 필요합니다.",
        "비정상입니다. 병변이 의심됩니다.",
        "비정상 X-ray입니다.",
        "비정상입니다.",
    ]
}

print("=" * 60)
print("X-ray 데이터 전처리 - MedGemma 형식 변환")
print("=" * 60)

def create_dataset(subset_name):
    """Train 또는 Val 데이터셋 생성"""
    subset_dir = DATA_DIR / subset_name

    data = []

    for class_name in ["Normal", "Abnormal"]:
        class_dir = subset_dir / class_name
        image_files = list(class_dir.glob("*.png"))

        print(f"\n{subset_name}/{class_name}: {len(image_files)}개 파일 처리 중...")

        for img_file in tqdm(image_files):
            # 이미지 상대 경로
            relative_path = f"{subset_name}/{class_name}/{img_file.name}"

            # 프롬프트 랜덤 선택
            user_prompt = random.choice(USER_PROMPTS)
            assistant_response = random.choice(ASSISTANT_RESPONSES[LABEL_MAP[class_name]])

            item = {
                "id": f"{LABEL_MAP[class_name]}_{img_file.stem}",
                "image": relative_path,
                "label": LABEL_MAP[class_name],
                "conversations": [
                    {
                        "role": "user",
                        "content": user_prompt
                    },
                    {
                        "role": "assistant",
                        "content": assistant_response
                    }
                ]
            }

            data.append(item)

    # 셔플
    random.shuffle(data)

    return data

# Train 데이터 생성
print("\n[1단계] Train 데이터 생성")
train_data = create_dataset("train")
print(f"Train 데이터 생성 완료: {len(train_data)}개")

# Val 데이터 생성
print("\n[2단계] Val 데이터 생성")
val_data = create_dataset("val")
print(f"Val 데이터 생성 완료: {len(val_data)}개")

# Val을 Val과 Test로 분할 (50:50)
print("\n[3단계] Val/Test 분할")
random.shuffle(val_data)
split_idx = len(val_data) // 2
val_split = val_data[:split_idx]
test_split = val_data[split_idx:]

print(f"Val: {len(val_split)}개")
print(f"Test: {len(test_split)}개")

# 저장
print("\n[4단계] JSON 파일 저장")

with open(OUTPUT_DIR / "train.json", 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)
print(f"저장: {OUTPUT_DIR / 'train.json'}")

with open(OUTPUT_DIR / "val.json", 'w', encoding='utf-8') as f:
    json.dump(val_split, f, ensure_ascii=False, indent=2)
print(f"저장: {OUTPUT_DIR / 'val.json'}")

with open(OUTPUT_DIR / "test.json", 'w', encoding='utf-8') as f:
    json.dump(test_split, f, ensure_ascii=False, indent=2)
print(f"저장: {OUTPUT_DIR / 'test.json'}")

# 통계
print("\n" + "=" * 60)
print("데이터 전처리 완료!")
print("=" * 60)

def print_stats(data, name):
    normal_count = sum(1 for x in data if x['label'] == 'normal')
    abnormal_count = sum(1 for x in data if x['label'] == 'abnormal')
    print(f"\n{name}:")
    print(f"  총: {len(data)}개")
    print(f"  Normal: {normal_count}개 ({normal_count/len(data)*100:.1f}%)")
    print(f"  Abnormal: {abnormal_count}개 ({abnormal_count/len(data)*100:.1f}%)")

print_stats(train_data, "Train")
print_stats(val_split, "Val")
print_stats(test_split, "Test")

print(f"\n✅ 전처리 완료!")
print(f"📁 출력 디렉토리: {OUTPUT_DIR}")
print(f"\n다음 단계: train_medgemma_lora_xray.py 실행")
