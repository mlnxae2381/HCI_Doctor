"""
MedGemma 4B Multimodal 테스트 스크립트
- 공식 Model Card 예제 기반
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

from huggingface_hub import login
login(token=os.environ.get("HF_TOKEN", ""))

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import requests

# ============================================
# 1. 모델 로드
# ============================================
print("=" * 60)
print("MedGemma 4B Multimodal 로딩 중...")
print("=" * 60)

model_id = "google/medgemma-4b-it"

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

print("모델 로드 완료!")
print()

# ============================================
# 2. 텍스트 전용 테스트 (의료 Q&A)
# ============================================
print("=" * 60)
print("[테스트 1] 의료 Q&A - 텍스트 전용")
print("=" * 60)

questions = [
    "폐렴의 주요 증상과 치료법에 대해 설명해주세요.",
    "당뇨병 환자가 주의해야 할 식이요법은 무엇인가요?",
]

for i, question in enumerate(questions, 1):
    print(f"\n질문 {i}: {question}")
    print("-" * 50)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=300, do_sample=False)
        generation = generation[0][input_len:]

    response = processor.decode(generation, skip_special_tokens=True)
    print(f"답변: {response}")
    print()

# ============================================
# 3. 멀티모달 테스트 - 흉부 X-ray 분석
# ============================================
print("=" * 60)
print("[테스트 2] 멀티모달 - 흉부 X-ray 분석")
print("=" * 60)

# 공식 예제의 이미지 URL (User-Agent 헤더 필요)
image_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
print(f"\n이미지 URL: {image_url}")

try:
    image = Image.open(requests.get(image_url, headers={"User-Agent": "example"}, stream=True).raw)
    print(f"이미지 로드 성공! 크기: {image.size}")

    # 방사선 전문의 역할로 분석
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert radiologist."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this X-ray in detail. What findings do you observe?"},
                {"type": "image", "image": image}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=500, do_sample=False)
        generation = generation[0][input_len:]

    response = processor.decode(generation, skip_special_tokens=True)
    print(f"\n[X-ray 분석 결과]")
    print("-" * 50)
    print(response)
    print()

except Exception as e:
    print(f"이미지 테스트 실패: {e}")

# ============================================
# 4. 대화형 모드
# ============================================
print("=" * 60)
print("[테스트 3] 대화형 모드")
print("=" * 60)
print("'quit' 입력 시 종료")
print("'image:URL' 형식으로 이미지 분석 가능")
print()

while True:
    user_input = input("질문: ").strip()

    if user_input.lower() == 'quit':
        print("테스트 종료!")
        break

    if not user_input:
        continue

    # 이미지 URL이 포함된 경우
    if user_input.startswith("image:"):
        parts = user_input.split(" ", 1)
        img_url = parts[0].replace("image:", "")
        question = parts[1] if len(parts) > 1 else "Describe this image."

        try:
            img = Image.open(requests.get(img_url, headers={"User-Agent": "example"}, stream=True).raw)
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an expert medical professional."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image", "image": img}
                    ]
                }
            ]
        except Exception as e:
            print(f"이미지 로드 실패: {e}")
            continue
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_input}
                ]
            }
        ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=500, do_sample=False)
        generation = generation[0][input_len:]

    response = processor.decode(generation, skip_special_tokens=True)
    print(f"MedGemma: {response}")
    print()
