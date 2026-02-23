"""
CT 폐암 데이터 전처리 및 증강 스크립트
- Train/Val/Test 분할
- 클래스 불균형 해결을 위한 데이터 증강
- MedGemma 파인튜닝용 JSON 포맷 생성
"""

import os
import json
import random
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from sklearn.model_selection import train_test_split

# 시드 고정
random.seed(42)
np.random.seed(42)

# 경로 설정
BASE_DIR = Path("/home/students/cs/doctor/AI_Doctor/data")
RAW_DIR = BASE_DIR / "Lung Cancer Dataset" / "The IQ-OTHNCCD lung cancer dataset" / "The IQ-OTHNCCD lung cancer dataset"
PROCESSED_DIR = BASE_DIR / "processed"

# 클래스 매핑 (폴더명 -> 라벨)
CLASS_MAPPING = {
    "Normal cases": "normal",
    "Bengin cases": "benign",  # 원본 폴더명 오타 그대로 사용
    "Malignant cases": "malignant"
}

# 목표 이미지 수 (클래스당)
TARGET_COUNT = 600


def load_images_by_class():
    """클래스별 이미지 경로 로드"""
    images_by_class = {}

    for folder_name, label in CLASS_MAPPING.items():
        folder_path = RAW_DIR / folder_name
        if folder_path.exists():
            # jpg, jpeg, png 모두 지원
            images = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
                images.extend(list(folder_path.glob(ext)))
            images_by_class[label] = images
            print(f"{label}: {len(images)}장 로드됨")
        else:
            print(f"경고: {folder_path} 폴더를 찾을 수 없습니다.")

    return images_by_class


def split_data(images_by_class, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Train/Val/Test 분할"""
    splits = {"train": {}, "val": {}, "test": {}}

    for label, images in images_by_class.items():
        # 먼저 train과 나머지로 분할
        train_imgs, temp_imgs = train_test_split(
            images, train_size=train_ratio, random_state=42
        )

        # 나머지를 val과 test로 분할
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_imgs, test_imgs = train_test_split(
            temp_imgs, train_size=val_ratio_adjusted, random_state=42
        )

        splits["train"][label] = train_imgs
        splits["val"][label] = val_imgs
        splits["test"][label] = test_imgs

        print(f"{label}: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

    return splits


def augment_image(image, augment_type):
    """이미지 증강 함수"""
    if augment_type == "rotate_90":
        return image.rotate(90)
    elif augment_type == "rotate_180":
        return image.rotate(180)
    elif augment_type == "rotate_270":
        return image.rotate(270)
    elif augment_type == "flip_horizontal":
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    elif augment_type == "flip_vertical":
        return image.transpose(Image.FLIP_TOP_BOTTOM)
    elif augment_type == "brightness_up":
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(1.2)
    elif augment_type == "brightness_down":
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(0.8)
    elif augment_type == "contrast_up":
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(1.2)
    elif augment_type == "contrast_down":
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(0.8)
    else:
        return image


def process_and_augment(splits):
    """데이터 증강 및 저장"""
    augment_types = [
        "rotate_90", "rotate_180", "rotate_270",
        "flip_horizontal", "flip_vertical",
        "brightness_up", "brightness_down",
        "contrast_up", "contrast_down"
    ]

    processed_counts = {"train": {}, "val": {}, "test": {}}

    for split_name, class_images in splits.items():
        print(f"\n=== {split_name} 세트 처리 중 ===")

        for label, images in class_images.items():
            output_dir = PROCESSED_DIR / split_name / label
            output_dir.mkdir(parents=True, exist_ok=True)

            count = 0

            # 원본 이미지 복사
            for img_path in images:
                img = Image.open(img_path)
                # RGB로 변환 (일부 이미지가 그레이스케일일 수 있음)
                if img.mode != "RGB":
                    img = img.convert("RGB")

                output_path = output_dir / f"{label}_{count:04d}.png"
                img.save(output_path)
                count += 1

            # Train 세트만 증강 (Val/Test는 원본 유지)
            if split_name == "train":
                original_count = count
                target = TARGET_COUNT

                # 클래스별 목표 수에 맞게 증강
                while count < target:
                    # 랜덤하게 원본 이미지 선택
                    img_path = random.choice(images)
                    img = Image.open(img_path)

                    if img.mode != "RGB":
                        img = img.convert("RGB")

                    # 랜덤하게 증강 기법 선택
                    aug_type = random.choice(augment_types)
                    augmented_img = augment_image(img, aug_type)

                    output_path = output_dir / f"{label}_{count:04d}_aug_{aug_type}.png"
                    augmented_img.save(output_path)
                    count += 1

                print(f"  {label}: {original_count}장 → {count}장 (증강 완료)")
            else:
                print(f"  {label}: {count}장 (원본 유지)")

            processed_counts[split_name][label] = count

    return processed_counts


def create_medgemma_json(processed_counts):
    """MedGemma 파인튜닝용 JSON 파일 생성"""

    # 클래스별 응답 템플릿
    response_templates = {
        "normal": [
            "이 CT 이미지는 정상 소견입니다. 폐 실질에 특이 병변이 관찰되지 않으며, 기관지와 혈관 구조가 정상적으로 보입니다.",
            "정상 폐 CT 소견입니다. 종양이나 결절 등의 이상 소견이 없습니다.",
            "이 흉부 CT 스캔은 정상입니다. 폐암을 시사하는 소견이 관찰되지 않습니다."
        ],
        "benign": [
            "이 CT 이미지에서 양성 병변이 관찰됩니다. 악성 종양의 특징은 보이지 않으나, 추가적인 경과 관찰이 권장됩니다.",
            "양성 폐 결절 소견입니다. 경계가 명확하고 석회화 소견이 있어 양성 병변으로 판단됩니다.",
            "이 CT에서 양성 종양이 의심됩니다. 크기와 형태로 보아 악성 가능성은 낮습니다."
        ],
        "malignant": [
            "이 CT 이미지에서 악성 종양이 의심됩니다. 불규칙한 경계와 주변 조직 침윤 소견이 관찰되며, 조직 검사가 필요합니다.",
            "폐암 의심 소견입니다. 종괴의 크기, 형태, 주변 구조물과의 관계를 고려할 때 악성 가능성이 높습니다.",
            "악성 폐 병변이 관찰됩니다. 추가적인 PET-CT 및 조직 검사를 통한 확진이 권장됩니다."
        ]
    }

    # 질문 템플릿
    question_templates = [
        "이 CT 이미지를 분석하여 폐암 여부를 판단해주세요.",
        "이 흉부 CT 스캔에서 이상 소견이 있는지 분석해주세요.",
        "이 CT 이미지에서 폐 병변이 관찰되는지 평가해주세요.",
        "이 폐 CT 영상을 보고 정상/양성/악성 여부를 판단해주세요."
    ]

    for split_name in ["train", "val", "test"]:
        data = []
        split_dir = PROCESSED_DIR / split_name

        for label in ["normal", "benign", "malignant"]:
            label_dir = split_dir / label
            if not label_dir.exists():
                continue

            for img_path in label_dir.glob("*.png"):
                # 상대 경로로 저장
                relative_path = str(img_path.relative_to(PROCESSED_DIR))

                entry = {
                    "id": img_path.stem,
                    "image": relative_path,
                    "label": label,
                    "conversations": [
                        {
                            "role": "user",
                            "content": random.choice(question_templates)
                        },
                        {
                            "role": "assistant",
                            "content": random.choice(response_templates[label])
                        }
                    ]
                }
                data.append(entry)

        # 셔플
        random.shuffle(data)

        # JSON 저장
        output_path = PROCESSED_DIR / f"{split_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"{split_name}.json 생성 완료: {len(data)}개 샘플")


def main():
    print("=" * 60)
    print("CT 폐암 데이터 전처리 시작")
    print("=" * 60)

    # 1. 이미지 로드
    print("\n[1단계] 이미지 로드")
    images_by_class = load_images_by_class()

    # 2. 데이터 분할
    print("\n[2단계] Train/Val/Test 분할")
    splits = split_data(images_by_class)

    # 3. 증강 및 저장
    print("\n[3단계] 데이터 증강 및 저장")
    processed_counts = process_and_augment(splits)

    # 4. MedGemma JSON 생성
    print("\n[4단계] MedGemma 파인튜닝용 JSON 생성")
    create_medgemma_json(processed_counts)

    # 결과 요약
    print("\n" + "=" * 60)
    print("처리 완료 요약")
    print("=" * 60)

    for split_name, counts in processed_counts.items():
        total = sum(counts.values())
        print(f"\n{split_name}: 총 {total}장")
        for label, count in counts.items():
            print(f"  - {label}: {count}장")

    print(f"\n출력 디렉토리: {PROCESSED_DIR}")
    print("JSON 파일: train.json, val.json, test.json")


if __name__ == "__main__":
    main()
