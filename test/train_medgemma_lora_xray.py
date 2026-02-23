"""
MedGemma X-ray LoRA 파인튜닝 스크립트
CT-LoRA와 동일한 방식으로 X-ray 데이터 학습
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU 1번만 사용 (Multi-GPU 오류 방지)

import json
import torch
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset
import numpy as np

# .env 로드
load_dotenv()

# 경로 설정
BASE_DIR = Path("/home/students/cs/doctor/AI_Doctor/data/AI_Doctor_Data/Chest_Xray/final_dataset")
DATA_DIR = Path("/home/students/cs/doctor/AI_Doctor/data/xray_processed")
OUTPUT_DIR = Path("/home/students/cs/doctor/AI_Doctor/test/xray_lora_results")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_ID = "google/medgemma-1.5-4b-it"

# 클래스 매핑
LABEL_MAP = {"normal": 0, "abnormal": 1}
LABEL_NAMES = ["normal", "abnormal"]

print("=" * 60)
print("MedGemma X-ray LoRA 파인튜닝")
print("=" * 60)

# ============================================
# 1. HuggingFace 로그인 및 모델 로드
# ============================================
print("\n[1단계] 모델 로드")

from huggingface_hub import login
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN이 설정되지 않았습니다!")

print(f"HuggingFace 로그인 중... (토큰: {HF_TOKEN[:10]}...)")
login(token=HF_TOKEN)

# Processor 로드
print("Processor 로드 중...")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# 베이스 모델 로드
print("베이스 모델 로드 중...")
base_model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

print("모델 로드 완료!")

# ============================================
# 2. LoRA 설정 (CT와 동일)
# ============================================
print("\n[2단계] LoRA 설정")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none"
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# Gradient checkpointing
model.enable_input_require_grads()
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()

# ============================================
# 3. 데이터셋 클래스
# ============================================
print("\n[3단계] 데이터셋 로드")

class XrayDataset(Dataset):
    def __init__(self, json_path, processor, base_dir):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.processor = processor
        self.base_dir = base_dir
        self.image_token = processor.tokenizer.boi_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 이미지 로드
        image_path = self.base_dir / item['image']
        image = Image.open(image_path).convert('RGB')

        # 대화 구성
        user_message = item['conversations'][0]['content']
        assistant_message = item['conversations'][1]['content']

        # 프롬프트 구성 (boi_token 포함)
        prompt = f"{self.image_token}{user_message}"

        # Processor로 인코딩
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )

        input_ids = inputs.input_ids[0]
        attention_mask = inputs.attention_mask[0]
        pixel_values = inputs.pixel_values[0]

        # token_type_ids 생성
        if 'token_type_ids' in inputs:
            token_type_ids = inputs.token_type_ids[0]
        else:
            token_type_ids = torch.zeros_like(input_ids)

        # 응답 토큰화
        assistant_ids = self.processor.tokenizer(
            assistant_message,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids[0]

        # EOS 토큰 추가
        eos_token_id = self.processor.tokenizer.eos_token_id
        assistant_ids = torch.cat([assistant_ids, torch.tensor([eos_token_id])])

        # 전체 시퀀스 구성
        input_ids_full = torch.cat([input_ids, assistant_ids])
        attention_mask_full = torch.cat([attention_mask, torch.ones(len(assistant_ids), dtype=torch.long)])
        token_type_ids_full = torch.cat([token_type_ids, torch.zeros(len(assistant_ids), dtype=torch.long)])

        # 라벨 생성 (프롬프트 부분은 -100)
        labels_full = input_ids_full.clone()
        labels_full[:len(input_ids)] = -100

        return {
            "input_ids": input_ids_full,
            "attention_mask": attention_mask_full,
            "token_type_ids": token_type_ids_full,
            "pixel_values": pixel_values,
            "labels": labels_full
        }

# 데이터셋 로드
train_dataset = XrayDataset(DATA_DIR / "train.json", processor, BASE_DIR)
val_dataset = XrayDataset(DATA_DIR / "val.json", processor, BASE_DIR)

print(f"Train 데이터: {len(train_dataset)}개")
print(f"Val 데이터: {len(val_dataset)}개")

# ============================================
# 4. Data Collator
# ============================================
def collate_fn(batch):
    """배치 데이터 패딩"""
    max_length = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    attention_mask = []
    token_type_ids = []
    pixel_values = []
    labels = []

    for item in batch:
        # 패딩
        pad_length = max_length - len(item["input_ids"])

        input_ids.append(torch.cat([
            item["input_ids"],
            torch.full((pad_length,), processor.tokenizer.pad_token_id, dtype=torch.long)
        ]))

        attention_mask.append(torch.cat([
            item["attention_mask"],
            torch.zeros(pad_length, dtype=torch.long)
        ]))

        token_type_ids.append(torch.cat([
            item["token_type_ids"],
            torch.zeros(pad_length, dtype=torch.long)
        ]))

        labels.append(torch.cat([
            item["labels"],
            torch.full((pad_length,), -100, dtype=torch.long)
        ]))

        pixel_values.append(item["pixel_values"])

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "token_type_ids": torch.stack(token_type_ids),
        "pixel_values": torch.stack(pixel_values),
        "labels": torch.stack(labels)
    }

# ============================================
# 5. 학습 설정
# ============================================
print("\n[4단계] 학습 설정")

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR / "checkpoints"),
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=64,
    learning_rate=5e-5,
    warmup_steps=100,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    bf16=True,
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    remove_unused_columns=False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
)

# ============================================
# 6. 학습 시작
# ============================================
print("\n[5단계] 학습 시작")
print("=" * 60)

trainer.train()

print("\n학습 완료!")

# ============================================
# 7. 모델 저장
# ============================================
print("\n[6단계] 모델 저장")

adapter_output = OUTPUT_DIR / "xray_lora_adapter"
model.save_pretrained(adapter_output)
processor.save_pretrained(adapter_output)

print(f"✅ LoRA 어댑터 저장: {adapter_output}")

# 학습 히스토리 저장
history = trainer.state.log_history
with open(OUTPUT_DIR / "training_history.json", 'w') as f:
    json.dump(history, f, indent=2)

print(f"✅ 학습 히스토리 저장: {OUTPUT_DIR / 'training_history.json'}")

print("\n" + "=" * 60)
print("MedGemma X-ray LoRA 파인튜닝 완료!")
print("=" * 60)
print(f"📁 출력 디렉토리: {OUTPUT_DIR}")
print(f"🎯 다음 단계: test_medgemma_lora_xray.py로 테스트")
