"""
MedGemma 1.5 4B CT-LoRA 파인튜닝 (Single GPU)
- CT 폐암 3클래스 분류: Normal, Benign, Malignant
- CT-LoRA (Low-Rank Adaptation for CT Classification) 사용
- 단일 GPU 학습 (GPU 0 사용)

실행 방법:
  python train_medgemma_lora.py
"""

import os
# 단일 GPU 사용 (python으로 실행)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU 0번만 사용

import json
import torch
from pathlib import Path
from PIL import Image
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드
load_dotenv()

from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 경로 설정
BASE_DIR = Path("/home/students/cs/doctor/AI_Doctor/data")
DATA_DIR = BASE_DIR / "processed"
OUTPUT_DIR = Path("/home/students/cs/doctor/AI_Doctor/test/ct_lora_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 모델 설정
MODEL_ID = "google/medgemma-1.5-4b-it"

# 클래스 매핑
LABEL_MAP = {"normal": 0, "benign": 1, "malignant": 2}
LABEL_NAMES = ["normal", "benign", "malignant"]

print("=" * 60)
print("MedGemma 1.5 4B CT-LoRA 파인튜닝 (Single GPU)")
print("=" * 60)


# ============================================
# 1. 데이터셋 클래스
# ============================================
class CTDataset(Dataset):
    """CT 이미지 + 텍스트 데이터셋"""

    def __init__(self, json_path: Path, data_dir: Path, processor):
        self.data_dir = data_dir
        self.processor = processor

        # JSON 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        print(f"데이터 로드: {len(self.data)}개 샘플")

        # 클래스별 개수 출력
        label_counts = {}
        for item in self.data:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1

        for label, count in label_counts.items():
            print(f"  - {label}: {count}개")

        # 이미지 토큰 설정 (Gemma3는 boi_token 사용)
        self.image_token = processor.tokenizer.boi_token  # <start_of_image>

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 이미지 로드
        image_path = self.data_dir / item['image']
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')

        # 대화 형식 구성
        conversations = item['conversations']
        user_message = conversations[0]['content']
        assistant_message = conversations[1]['content']

        # 프롬프트 구성: 이미지 토큰을 명시적으로 포함
        prompt = f"{self.image_token}{user_message}"

        # 1. 입력(이미지+질문) 인코딩
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="longest",
            truncation=True
        )

        # 2. 답변 인코딩
        labels = self.processor.tokenizer(
            text=assistant_message,
            return_tensors="pt",
            padding="longest",
            truncation=True
        ).input_ids

        input_ids = inputs.input_ids[0]
        attention_mask = inputs.attention_mask[0]
        pixel_values = inputs.pixel_values[0]
        label_ids = labels[0]

        # token_type_ids 가져오기 (Gemma3 필수)
        if 'token_type_ids' in inputs:
            token_type_ids = inputs.token_type_ids[0]
        else:
            # token_type_ids가 없으면 생성 (0: text, 1: image)
            token_type_ids = torch.zeros_like(input_ids)

        # EOS 토큰 추가
        if self.processor.tokenizer.eos_token_id is not None:
            label_ids = torch.cat([label_ids, torch.tensor([self.processor.tokenizer.eos_token_id])])

        # 3. 결합 (프롬프트 + 답변)
        input_ids_full = torch.cat([input_ids, label_ids])
        attention_mask_full = torch.cat([attention_mask, torch.ones_like(label_ids)])
        token_type_ids_full = torch.cat([token_type_ids, torch.zeros_like(label_ids)])  # 답변도 text로

        # Labels: 질문 부분은 -100으로 마스킹
        labels_full = torch.cat([
            torch.full_like(input_ids, -100),  # 질문+이미지 부분 마스킹
            label_ids                          # 답변 부분만 학습
        ])

        # Max Length Truncation
        MAX_LEN = 512
        if len(input_ids_full) > MAX_LEN:
            input_ids_full = input_ids_full[:MAX_LEN]
            attention_mask_full = attention_mask_full[:MAX_LEN]
            token_type_ids_full = token_type_ids_full[:MAX_LEN]
            labels_full = labels_full[:MAX_LEN]

        return {
            "input_ids": input_ids_full,
            "attention_mask": attention_mask_full,
            "token_type_ids": token_type_ids_full,
            "pixel_values": pixel_values,
            "labels": labels_full
        }


# Data Collator (배치 처리)
def collate_fn(batch):
    """배치 데이터를 적절히 패딩"""
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    pixel_values = torch.stack([item['pixel_values'] for item in batch])

    # Pad sequences
    from torch.nn.utils.rnn import pad_sequence

    # Padding value
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    token_type_ids_padded = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "token_type_ids": token_type_ids_padded,
        "pixel_values": pixel_values,
        "labels": labels_padded
    }


# ============================================
# 2. 모델 및 프로세서 로드
# ============================================
print("\n[1단계] MedGemma 1.5 4B 모델 로드")

from huggingface_hub import login

# .env 파일에서 HuggingFace 토큰 읽기
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN이 설정되지 않았습니다!\n"
        ".env 파일에 HF_TOKEN을 입력하세요.\n"
        "예: HF_TOKEN=hf_your_token_here"
    )

print(f"HuggingFace 로그인 중... (토큰: {HF_TOKEN[:10]}...)")
login(token=HF_TOKEN)

# Processor 로드
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# Pad token 설정 (없으면 eos_token 사용)
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    print("Pad token이 없어서 EOS token으로 설정했습니다.")

# 이미지 토큰 확인
print(f"\n토큰 정보:")
print(f"  - BOI token: {processor.tokenizer.boi_token} (ID: {processor.tokenizer.boi_token_id})")
print(f"  - Image token: {processor.tokenizer.image_token} (ID: {processor.tokenizer.image_token_id})")
print(f"  - EOI token: {processor.tokenizer.eoi_token} (ID: {processor.tokenizer.eoi_token_id})")
print(f"  - EOS token: {processor.tokenizer.eos_token} (ID: {processor.tokenizer.eos_token_id})")
print(f"  - PAD token: {processor.tokenizer.pad_token} (ID: {processor.tokenizer.pad_token_id})")
print(f"  - Vocab size: {len(processor.tokenizer)}")

# 모델 로드
print("모델 다운로드 및 로드 중...")
model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,  # config에서 bfloat16 사용
    # device_map="auto" 제거: DDP 환경에서 Trainer가 자동으로 GPU에 할당
    trust_remote_code=True  # 커스텀 모델 코드 신뢰
)

print(f"모델 로드 완료!")
print(f"  - 파라미터 수: {model.num_parameters() / 1e9:.2f}B")

# Gradient Checkpointing 활성화 (메모리 절약)
model.gradient_checkpointing_enable()


# ============================================
# 3. CT-LoRA 설정
# ============================================
print("\n[2단계] CT-LoRA 설정")

lora_config = LoraConfig(
    r=16,                      # LoRA rank
    lora_alpha=32,             # LoRA alpha
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Attention + MLP
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# CT-LoRA 적용
model = get_peft_model(model, lora_config)

print(f"CT-LoRA 적용 완료!")
model.print_trainable_parameters()


# ============================================
# 4. 데이터셋 로드
# ============================================
print("\n[3단계] 데이터셋 로드")

train_dataset = CTDataset(DATA_DIR / "train.json", DATA_DIR, processor)
val_dataset = CTDataset(DATA_DIR / "val.json", DATA_DIR, processor)
test_dataset = CTDataset(DATA_DIR / "test.json", DATA_DIR, processor)


# ============================================
# 5. 학습 설정
# ============================================
print("\n[4단계] 학습 설정")

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR / "checkpoints"),
    num_train_epochs=5,
    per_device_train_batch_size=1,      # 단일 GPU 사용
    per_device_eval_batch_size=1,       # 평가 시에도 동일
    gradient_accumulation_steps=64,     # 전체 배치 크기 64 유지 (1*1*64=64)
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    save_steps=200,
    eval_steps=200,
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=True,                           # BFloat16 사용 (L4 지원)
    dataloader_num_workers=4,
    remove_unused_columns=False,
    report_to="none",                    # wandb 비활성화
    ddp_find_unused_parameters=False,    # DDP 최적화
    gradient_checkpointing=True,         # 필수 추가: 메모리 사용량 대폭 감소
    gradient_checkpointing_kwargs={"use_reentrant": False},  # 호환성 옵션
)

print(f"\n학습 설정:")
print(f"  - GPU: 단일 GPU (GPU 0)")
print(f"  - Epochs: {training_args.num_train_epochs}")
print(f"  - 배치 크기 (per GPU): {training_args.per_device_train_batch_size}")
print(f"  - Gradient Accumulation Steps: {training_args.gradient_accumulation_steps}")
print(f"  - 실질 배치 크기: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  - Learning rate: {training_args.learning_rate}")
print(f"  - Gradient Checkpointing: Enabled")
print(f"  - BF16: {training_args.bf16}")


# ============================================
# 6. Trainer 및 학습
# ============================================
print("\n[5단계] 학습 시작")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,  # 커스텀 collator 사용
)

# 학습 실행
print("\n학습 시작...")
trainer.train()

print("\n학습 완료!")


# ============================================
# 7. 모델 저장
# ============================================
print("\n[6단계] 모델 저장")

# CT-LoRA 어댑터만 저장
model.save_pretrained(OUTPUT_DIR / "ct_lora_adapter")
processor.save_pretrained(OUTPUT_DIR / "ct_lora_adapter")

print(f"CT-LoRA 어댑터 저장: {OUTPUT_DIR / 'ct_lora_adapter'}")


# ============================================
# 8. 테스트 세트 평가
# ============================================
print("\n[7단계] 테스트 세트 평가")

# 추론 모드
model.eval()

predictions = []
true_labels = []

print("테스트 데이터 추론 중...")

for i in range(len(test_dataset)):
    # 이미지 로드
    image_path = DATA_DIR / test_dataset.data[i]['image']
    image = Image.open(image_path).convert('RGB')

    # 실제 라벨 가져오기
    true_label = LABEL_MAP[test_dataset.data[i]['label']]

    # 프롬프트 (boi_token 포함)
    boi_token = processor.tokenizer.boi_token
    prompt = f"{boi_token}{test_dataset.data[i]['conversations'][0]['content']}"

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
            do_sample=False
        )

    # 디코딩
    generated_text = processor.decode(outputs[0], skip_special_tokens=True)

    # 라벨 추출 (간단한 휴리스틱)
    generated_lower = generated_text.lower()
    if "malignant" in generated_lower or "악성" in generated_lower:
        pred_label = 2
    elif "benign" in generated_lower or "양성" in generated_lower:
        pred_label = 1
    else:
        pred_label = 0  # normal

    predictions.append(pred_label)
    true_labels.append(true_label)

    if (i + 1) % 20 == 0:
        print(f"  진행: {i+1}/{len(test_dataset)}")

# 평가 지표
accuracy = accuracy_score(true_labels, predictions)
print(f"\n테스트 정확도: {accuracy:.4f}")

print("\n분류 리포트:")
print(classification_report(true_labels, predictions, target_names=LABEL_NAMES))

# Confusion Matrix
cm = confusion_matrix(true_labels, predictions)
print("\nConfusion Matrix:")
print(cm)

# 시각화
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

# 결과 저장
results = {
    "model": MODEL_ID,
    "method": "CT-LoRA Fine-tuning",
    "test_accuracy": float(accuracy),
    "test_samples": len(test_dataset),
    "confusion_matrix": cm.tolist(),
    "classification_report": classification_report(true_labels, predictions, target_names=LABEL_NAMES, output_dict=True)
}

with open(OUTPUT_DIR / 'results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"결과 저장: {OUTPUT_DIR / 'results.json'}")

print("\n" + "=" * 60)
print("파인튜닝 완료!")
print("=" * 60)
print(f"테스트 정확도: {accuracy:.4f}")
print(f"결과 디렉토리: {OUTPUT_DIR}")
