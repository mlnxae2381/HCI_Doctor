"""
X-ray 이미지 분석 모델 (MedGemma LoRA)
정확도: 72.38% (테스트 11,213샘플)
분류: Normal / Abnormal
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
ADAPTER_PATH = Path("/home/students/cs/doctor/AI_Doctor/test/xray_lora_results/xray_lora_adapter")

LABEL_MAP = {"normal": 0, "abnormal": 1}
LABEL_NAMES = ["normal", "abnormal"]

class XrayModel:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_trained = ADAPTER_PATH.exists()

    def load(self):
        """모델 로드"""
        if not self.is_trained:
            raise Exception("X-ray 모델이 아직 학습 중입니다. 학습 완료까지 3-4일 소요 예정입니다.")

        if self.model is not None:
            return

        print("Loading X-ray Model (MedGemma LoRA)...")

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

        print(f"X-ray Model loaded on {self.device}")

    def extract_label(self, text):
        """개선된 라벨 추출 - 프롬프트 제거 후 응답 부분만 분석"""
        text_lower = text.lower().strip()

        # 프롬프트 이후 생성된 응답 부분만 추출
        # (프롬프트의 "이상 소견이 있는지" 같은 표현이 scoring을 오염시키는 것 방지)
        response_text = text_lower
        for ending in ['판단해주세요.', '평가해주세요.', '알려주세요.', '분석해주세요.']:
            idx = text_lower.find(ending)
            if idx != -1:
                response_text = text_lower[idx + len(ending):].strip()
                break

        # 1. 직접 키워드 체크 (명확한 판단)
        if any(phrase in response_text for phrase in ['비정상입니다', '비정상 소견', '비정상 x-ray']):
            return 1, {'normal': 0, 'abnormal': 15}

        if any(phrase in response_text for phrase in ['정상입니다', '정상 소견', '정상 x-ray', '정상 폐', '정상 범위']):
            return 0, {'normal': 10, 'abnormal': 0}

        # 2. 점수 계산 (응답 부분만)
        scores = {'normal': 0, 'abnormal': 0}

        if '정상' in response_text:
            scores['normal'] += 10

        if '이상 없' in response_text or '소견 없' in response_text:
            scores['normal'] += 10

        if '관찰되지 않' in response_text or '보이지 않' in response_text:
            scores['normal'] += 8

        if '병적 소견' in response_text and ('없' in response_text or '관찰되지' in response_text):
            scores['normal'] += 6

        if '비정상' in response_text:
            scores['abnormal'] += 15

        if '이상 소견이' in response_text or '이상소견이' in response_text:
            scores['abnormal'] += 12

        # '관찰되지 않'도 부정 표현으로 처리 (없 없어도 OK)
        if '이상 소견' in response_text and '없' not in response_text and '관찰되지' not in response_text:
            scores['abnormal'] += 10

        if '병변' in response_text:
            if '병변이 없' in response_text or '병변 없' in response_text:
                scores['normal'] += 5
            else:
                scores['abnormal'] += 8

        if '추가 검사' in response_text or '전문의' in response_text:
            scores['abnormal'] += 6

        if '추가 진료' in response_text or '추가 검진' in response_text:
            scores['abnormal'] += 4

        # 점수 기반 판단
        if scores['abnormal'] > scores['normal']:
            return 1, scores
        return 0, scores

    def predict(self, image_path):
        """X-ray 이미지 분석"""
        if not self.is_trained:
            raise Exception("X-ray 모델이 학습되지 않았습니다. xray_lora_adapter 디렉토리를 확인하세요.")

        if self.model is None:
            self.load()

        # 이미지 로드
        image = Image.open(image_path).convert('RGB')

        # 프롬프트
        boi_token = self.processor.tokenizer.boi_token
        user_message = "이 흉부 X-ray 이미지를 분석하여 이상 소견이 있는지 판단해주세요."
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
                max_new_tokens=100,
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
            confidence = {k: 0.5 for k in scores}

        return {
            'prediction': LABEL_NAMES[pred_label],
            'confidence': confidence,
            'explanation': generated_text,
            'model': 'MedGemma LoRA',
            'accuracy': '72.38%',
            'training_info': {
                'checkpoint': 'step-3600',
                'val_loss': 0.0929,
                'epochs': '2.6/3.0'
            }
        }

# 전역 모델 인스턴스
xray_model = XrayModel()
