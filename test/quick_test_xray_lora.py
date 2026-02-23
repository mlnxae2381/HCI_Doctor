"""
X-ray LoRA 모델 빠른 검증 테스트
모델 로드 및 샘플 이미지 1개로 추론 테스트
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU 1번 사용

import json
import torch
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from peft import PeftModel
import re

# .env 로드
load_dotenv()

# 경로 설정
BASE_DIR = Path("/home/students/cs/doctor/AI_Doctor/data/AI_Doctor_Data/Chest_Xray/final_dataset")
DATA_DIR = Path("/home/students/cs/doctor/AI_Doctor/data/xray_processed")
OUTPUT_DIR = Path("/home/students/cs/doctor/AI_Doctor/test/xray_lora_results")
ADAPTER_PATH = OUTPUT_DIR / "xray_lora_adapter"

MODEL_ID = "google/medgemma-1.5-4b-it"

LABEL_MAP = {"normal": 0, "abnormal": 1}
LABEL_NAMES = ["normal", "abnormal"]

print("=" * 60)
print("X-ray LoRA 빠른 검증 테스트")
print("=" * 60)

# HuggingFace 로그인
from huggingface_hub import login
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN이 설정되지 않았습니다!")

print(f"\n✅ HuggingFace 로그인 중...")
login(token=HF_TOKEN)

# 어댑터 존재 확인
if not ADAPTER_PATH.exists():
    raise FileNotFoundError(f"어댑터를 찾을 수 없습니다: {ADAPTER_PATH}")

print(f"✅ 어댑터 확인: {ADAPTER_PATH}")

# 모델 로드
print("\n[1단계] 모델 로드 중...")

print("  - Processor 로드...")
processor = AutoProcessor.from_pretrained(ADAPTER_PATH, trust_remote_code=True)

print("  - 베이스 모델 로드...")
base_model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

print("  - LoRA 어댑터 로드...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model = model.to("cuda")
model.eval()

print("✅ 모델 로드 완료!")

# 테스트 데이터에서 샘플 2개 로드
print("\n[2단계] 샘플 테스트 (2개)")

with open(DATA_DIR / "test.json", 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# Normal 1개, Abnormal 1개 선택
normal_sample = next(x for x in test_data if x['label'] == 'normal')
abnormal_sample = next(x for x in test_data if x['label'] == 'abnormal')

samples = [normal_sample, abnormal_sample]

def extract_label(text):
    """생성된 텍스트에서 라벨 추출"""
    text_lower = text.lower()
    scores = {'normal': 0, 'abnormal': 0}

    if '정상' in text_lower:
        if re.search(r'정상\s*(소견|범위|x-ray|영상)', text_lower):
            scores['normal'] += 6
        elif '정상입니다' in text_lower or '정상이다' in text_lower:
            scores['normal'] += 5
        else:
            scores['normal'] += 3

    if re.search(r'이상\s*소견.*없', text_lower) or re.search(r'특이\s*소견.*없', text_lower):
        scores['normal'] += 5

    if '비정상' in text_lower:
        scores['abnormal'] += 6

    if '이상 소견' in text_lower or '이상 음영' in text_lower:
        scores['abnormal'] += 6

    max_label = max(scores, key=scores.get)
    max_score = scores[max_label]

    if max_score == 0:
        return 0, scores

    return LABEL_MAP[max_label], scores

for idx, sample in enumerate(samples):
    print(f"\n--- 샘플 {idx+1} ---")

    # 이미지 로드
    image_path = BASE_DIR / sample['image']
    image = Image.open(image_path).convert('RGB')

    true_label = LABEL_MAP[sample['label']]
    print(f"실제 라벨: {sample['label']}")

    # 프롬프트
    boi_token = processor.tokenizer.boi_token
    user_message = sample['conversations'][0]['content']
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
    print("추론 중...")
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
    pred_label, scores = extract_label(generated_text)

    print(f"예측 라벨: {LABEL_NAMES[pred_label]}")
    print(f"점수: normal={scores['normal']}, abnormal={scores['abnormal']}")
    print(f"결과: {'✅ 정답' if true_label == pred_label else '❌ 오답'}")
    print(f"생성 텍스트: {generated_text[:150]}...")

print("\n" + "=" * 60)
print("✨ 빠른 검증 완료!")
print("=" * 60)
print("모델이 정상적으로 로드되고 추론이 가능합니다.")
print("\n다음 단계:")
print("  1. python test_medgemma_lora_xray.py  # 전체 테스트 세트 평가")
print("  2. 웹앱에서 실제 사용")
print("=" * 60)
