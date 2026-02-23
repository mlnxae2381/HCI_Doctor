"""
CT 이미지 분석 모델 (MedGemma LoRA)
정확도: 87.35%
분류: Normal / Benign / Malignant
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from peft import PeftModel
from pathlib import Path
import re

MODEL_ID = "google/medgemma-1.5-4b-it"
ADAPTER_PATH = Path("/home/students/cs/doctor/AI_Doctor/test/ct_lora_results/ct_lora_adapter")

LABEL_MAP = {"normal": 0, "benign": 1, "malignant": 2}
LABEL_NAMES = ["normal", "benign", "malignant"]

class CTModel:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        """모델 로드"""
        if self.model is not None:
            return

        print("Loading CT Model (MedGemma LoRA)...")

        # Processor 로드
        self.processor = AutoProcessor.from_pretrained(ADAPTER_PATH, trust_remote_code=True)

        # 베이스 모델 로드
        base_model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # LoRA 어댑터 로드
        self.model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"CT Model loaded on {self.device}")

    def extract_label(self, text):
        """생성된 텍스트에서 라벨 추출 (프롬프트 오염 방지)"""
        text_lower = text.lower().strip()

        # 프롬프트 이후 응답 부분만 추출
        rt = text_lower
        for ending in ['판단해주세요.', '평가해주세요.', '알려주세요.', '분석해주세요.']:
            idx = text_lower.find(ending)
            if idx != -1:
                rt = text_lower[idx + len(ending):].strip()
                break

        scores = {'normal': 0, 'benign': 0, 'malignant': 0}

        # Normal 시그널
        if '정상' in rt:
            if re.search(r'정상\s*(소견|범위|ct|영상)', rt):
                scores['normal'] += 6
            elif '정상입니다' in rt or '정상이다' in rt:
                scores['normal'] += 5
            else:
                scores['normal'] += 3

        if re.search(r'이상\s*소견.*없', rt) or re.search(r'특이\s*소견.*없', rt):
            scores['normal'] += 5
        if '관찰되지 않' in rt or '보이지 않' in rt:
            scores['normal'] += 4

        # Benign 시그널
        if '양성' in rt:
            if re.search(r'양성\s*(종양|병변|결절)', rt):
                scores['benign'] += 6
            else:
                scores['benign'] += 4

        if re.search(r'악성\s*가능성.*낮', rt):
            scores['benign'] += 4

        # Malignant 시그널
        has_negation = bool(re.search(r'악성.*아니|악성.*없|악성.*낮', rt))
        if '악성' in rt and not has_negation:
            if re.search(r'악성\s*(종양|암|병변)', rt):
                scores['malignant'] += 7
            else:
                scores['malignant'] += 5

        if '암' in rt and not re.search(r'암.*아니|암.*없|암.*관찰되지', rt):
            scores['malignant'] += 6

        # 최고 점수 라벨 반환
        max_label = max(scores, key=scores.get)
        max_score = scores[max_label]

        if max_score == 0:
            return 0, scores  # 기본값 normal

        return LABEL_MAP[max_label], scores

    def predict(self, image_path):
        """CT 이미지 분석"""
        if self.model is None:
            self.load()

        # 이미지 로드
        image = Image.open(image_path).convert('RGB')

        # 프롬프트
        boi_token = self.processor.tokenizer.boi_token
        user_message = "이 CT 이미지를 분석하여 병변이 정상인지, 양성 종양인지, 악성 종양인지 판단해주세요."
        prompt = f"{boi_token}{user_message}"

        # 인코딩
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        )

        if 'token_type_ids' not in inputs:
            inputs['token_type_ids'] = torch.zeros_like(inputs['input_ids'])

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )

        # 디코딩
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)

        # 라벨 추출
        pred_label, scores = self.extract_label(generated_text)

        # 점수를 0~1 확률로 정규화
        total = sum(scores.values())
        if total > 0:
            confidence = {k: round(v / total, 4) for k, v in scores.items()}
        else:
            confidence = {k: 1/3 for k in scores}

        return {
            'prediction': LABEL_NAMES[pred_label],
            'confidence': confidence,
            'explanation': generated_text,
            'model': 'MedGemma LoRA',
            'accuracy': '87.35%'
        }

# 전역 모델 인스턴스
ct_model = CTModel()
