"""
오디오 분석 모델 (HeAR - Health Acoustic Representations)
정확도: 96.74%
분류: Normal / Abnormal (호흡기 질환)
용도: 기침음 & 호흡음 분석
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
from pathlib import Path
from huggingface_hub import snapshot_download
from sklearn.linear_model import LogisticRegression
import librosa
import pickle

# TensorFlow 설정
tf.get_logger().setLevel('ERROR')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU 설정 오류: {e}")

TARGET_SR = 16000
SEGMENT_LENGTH = 2.0
SAMPLES_PER_SEGMENT = int(TARGET_SR * SEGMENT_LENGTH)

MODEL_PATH = Path("/home/students/cs/doctor/AI_Doctor/test/hear_general_results")

class AudioModel:
    def __init__(self):
        self.hear_model = None
        self.classifier = None

    def load(self):
        """모델 로드"""
        if self.hear_model is not None:
            return

        print("Loading HeAR Model for Audio Analysis...")

        # HeAR 모델 다운로드 및 로드
        model_id = "google/hear"
        model_path = snapshot_download(repo_id=model_id)
        self.hear_model = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')

        # 사전 학습된 분류기 로드
        classifier_path = MODEL_PATH / "classifier.pkl"
        if classifier_path.exists():
            with open(classifier_path, 'rb') as f:
                self.classifier = pickle.load(f)
            print("Pre-trained classifier loaded")
        else:
            # 기본 분류기 (학습 데이터가 있을 경우)
            self.classifier = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
            print("Using default classifier")

        print("HeAR Audio Model loaded")

    def preprocess_audio(self, audio_path):
        """오디오 전처리"""
        # 오디오 로드
        audio, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)

        # 2초로 자르기 또는 패딩
        if len(audio) < SAMPLES_PER_SEGMENT:
            audio = np.pad(audio, (0, SAMPLES_PER_SEGMENT - len(audio)))
        else:
            audio = audio[:SAMPLES_PER_SEGMENT]

        return audio

    def extract_embedding(self, audio):
        """HeAR 임베딩 추출"""
        audio_tensor = tf.constant([audio], dtype=tf.float32)
        outputs = self.hear_model(audio_tensor)
        emb_key = list(outputs.keys())[0]
        embedding = outputs[emb_key].numpy()[0]
        return embedding

    def predict(self, audio_path):
        """오디오 분석"""
        if self.hear_model is None:
            self.load()

        # 오디오 전처리
        audio = self.preprocess_audio(audio_path)

        # 임베딩 추출
        embedding = self.extract_embedding(audio)

        # 분류기가 학습되어 있으면 예측
        if hasattr(self.classifier, 'predict'):
            try:
                prediction = self.classifier.predict([embedding])[0]
                proba = self.classifier.predict_proba([embedding])[0]

                label = "abnormal" if prediction == 1 else "normal"
                confidence = {
                    "normal": float(proba[0]),
                    "abnormal": float(proba[1])
                }

                # 진단 설명 생성
                if label == "normal":
                    explanation = "정상적인 호흡음/기침 소리입니다. 폐 기능이 정상 범위 내에 있습니다."
                else:
                    explanation = f"비정상 소견이 감지되었습니다 (신뢰도: {confidence['abnormal']*100:.1f}%). 호흡기 질환 가능성이 있으니 전문의 상담을 권장합니다."

                return {
                    'prediction': label,
                    'confidence': confidence,
                    'explanation': explanation,
                    'model': 'HeAR',
                    'accuracy': '96.74%'
                }
            except Exception as e:
                # 분류기가 제대로 학습되지 않은 경우
                return {
                    'error': '분류기가 아직 학습되지 않았습니다',
                    'embedding_shape': embedding.shape,
                    'message': '임베딩은 추출되었으나 분류 모델이 필요합니다'
                }
        else:
            # 임베딩만 반환 (분류기가 없는 경우)
            return {
                'embedding_extracted': True,
                'embedding_shape': embedding.shape,
                'message': '임베딩 추출 성공 (분류기 학습 필요)'
            }

# 전역 모델 인스턴스
audio_model = AudioModel()
