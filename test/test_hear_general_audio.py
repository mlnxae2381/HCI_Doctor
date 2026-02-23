"""
HeAR 모델 - 일반 오디오 데이터 테스트 (Zero-shot)
COVID가 아닌 일반 정상/비정상 오디오로 HeAR의 일반화 성능 평가
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # GPU 2번 사용
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
from pathlib import Path
from huggingface_hub import snapshot_download, login
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import librosa
import json
from tqdm import tqdm
from dotenv import load_dotenv

# .env 로드
load_dotenv()

# TensorFlow GPU 설정
tf.get_logger().setLevel('ERROR')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU 설정 오류: {e}")

print("=" * 60)
print("HeAR - 일반 오디오 데이터 테스트 (Zero-shot)")
print("=" * 60)

# 경로 설정
DATA_DIR = Path("/home/students/cs/doctor/AI_Doctor/data/AI_Doctor_Data/classified_audio")
OUTPUT_DIR = Path("/home/students/cs/doctor/AI_Doctor/test/hear_general_results")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_SR = 16000  # HeAR 요구사항
SEGMENT_LENGTH = 2.0  # 2초
SAMPLES_PER_SEGMENT = int(TARGET_SR * SEGMENT_LENGTH)

# ============================================
# 1. HeAR 모델 로드
# ============================================
print("\n[1단계] HeAR 모델 로드")

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

model_id = "google/hear"
print(f"모델 다운로드: {model_id}")
model_path = snapshot_download(repo_id=model_id)

print("HeAR 모델 로딩 중...")
model = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
print("✅ HeAR 모델 로드 완료!")

# ============================================
# 2. 데이터 로드
# ============================================
print("\n[2단계] 오디오 데이터 로드")

def load_and_preprocess_audio(file_path):
    """오디오 파일 로드 및 전처리"""
    try:
        # 오디오 로드
        audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)

        # 2초로 자르기 또는 패딩
        if len(audio) < SAMPLES_PER_SEGMENT:
            # 패딩
            audio = np.pad(audio, (0, SAMPLES_PER_SEGMENT - len(audio)))
        else:
            # 앞쪽 2초만 사용
            audio = audio[:SAMPLES_PER_SEGMENT]

        return audio
    except Exception as e:
        print(f"오디오 로드 실패 {file_path}: {e}")
        return None

# Normal 파일 로드
normal_dir = DATA_DIR / "Normal"
abnormal_dir = DATA_DIR / "Abnormal"

normal_files = list(normal_dir.glob("*.wav"))
abnormal_files = list(abnormal_dir.glob("*.wav"))

print(f"Normal 파일: {len(normal_files)}개")
print(f"Abnormal 파일: {len(abnormal_files)}개")
print(f"비율: 1:{len(abnormal_files) / len(normal_files):.1f} (심각한 불균형)")

# 오디오 로드
print("\n오디오 로드 중...")
audio_data = []
labels = []

for file_path in tqdm(normal_files, desc="Normal"):
    audio = load_and_preprocess_audio(file_path)
    if audio is not None:
        audio_data.append(audio)
        labels.append(0)  # Normal

for file_path in tqdm(abnormal_files, desc="Abnormal"):
    audio = load_and_preprocess_audio(file_path)
    if audio is not None:
        audio_data.append(audio)
        labels.append(1)  # Abnormal

audio_data = np.array(audio_data, dtype=np.float32)
labels = np.array(labels)

print(f"\n총 샘플 수: {len(audio_data)}")
print(f"  Normal: {np.sum(labels == 0)}개")
print(f"  Abnormal: {np.sum(labels == 1)}개")

# ============================================
# 3. Train/Test Split
# ============================================
print("\n[3단계] Train/Test Split")

# Stratified split (클래스 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(
    audio_data, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Train: {len(X_train)}개 (Normal: {np.sum(y_train == 0)}, Abnormal: {np.sum(y_train == 1)})")
print(f"Test: {len(X_test)}개 (Normal: {np.sum(y_test == 0)}, Abnormal: {np.sum(y_test == 1)})")

# ============================================
# 4. HeAR 임베딩 추출
# ============================================
print("\n[4단계] HeAR 임베딩 추출")

def extract_embeddings(audio_batch, name="Dataset"):
    """HeAR 임베딩 추출"""
    embeddings = []

    batch_size = 32
    for i in tqdm(range(0, len(audio_batch), batch_size), desc=f"{name} 임베딩 추출"):
        batch = audio_batch[i:i+batch_size]
        batch_tensor = tf.constant(batch, dtype=tf.float32)

        outputs = model(batch_tensor)
        emb_key = list(outputs.keys())[0]
        batch_embeddings = outputs[emb_key].numpy()

        embeddings.append(batch_embeddings)

    return np.concatenate(embeddings, axis=0)

X_train_emb = extract_embeddings(X_train, "Train")
X_test_emb = extract_embeddings(X_test, "Test")

print(f"\n임베딩 shape:")
print(f"  Train: {X_train_emb.shape}")
print(f"  Test: {X_test_emb.shape}")

# 임베딩 저장
np.save(OUTPUT_DIR / "train_embeddings.npy", X_train_emb)
np.save(OUTPUT_DIR / "test_embeddings.npy", X_test_emb)
np.save(OUTPUT_DIR / "train_labels.npy", y_train)
np.save(OUTPUT_DIR / "test_labels.npy", y_test)

# ============================================
# 5. 분류 모델 학습 (Class Weighting)
# ============================================
print("\n[5단계] 로지스틱 회귀 학습 (Class Weighting)")

# Class weight 계산 (불균형 해결)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

print(f"Class weights: Normal={class_weights[0]:.2f}, Abnormal={class_weights[1]:.2f}")

clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
clf.fit(X_train_emb, y_train)

# ============================================
# 6. 평가
# ============================================
print("\n[6단계] 테스트 평가")

y_pred = clf.predict(X_test_emb)
y_pred_proba = clf.predict_proba(X_test_emb)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
cm = confusion_matrix(y_test, y_pred)

print(f"\n✨ 평가 결과:")
print(f"  정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  AUC-ROC: {auc:.4f}")

print("\n분류 리포트:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Abnormal"]))

print("\nConfusion Matrix:")
print(cm)
print(f"  Normal → Normal: {cm[0][0]} (정답)")
print(f"  Normal → Abnormal: {cm[0][1]} (오분류)")
print(f"  Abnormal → Normal: {cm[1][0]} (오분류)")
print(f"  Abnormal → Abnormal: {cm[1][1]} (정답)")

# ============================================
# 7. 결과 저장
# ============================================
print("\n[7단계] 결과 저장")

results = {
    "model": "google/hear",
    "data_source": "AI_Doctor_Data/classified_audio (일반 오디오)",
    "total_samples": int(len(labels)),
    "train_samples": int(len(y_train)),
    "test_samples": int(len(y_test)),
    "data_distribution": {
        "normal": int(np.sum(labels == 0)),
        "abnormal": int(np.sum(labels == 1)),
        "ratio": f"1:{len(abnormal_files) / len(normal_files):.1f}"
    },
    "test_accuracy": float(accuracy),
    "test_auc": float(auc),
    "confusion_matrix": cm.tolist(),
    "classification_report": classification_report(y_test, y_pred, target_names=["Normal", "Abnormal"], output_dict=True),
    "class_weights_used": {
        "normal": float(class_weights[0]),
        "abnormal": float(class_weights[1])
    }
}

with open(OUTPUT_DIR / "results.json", 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"결과 저장: {OUTPUT_DIR / 'results.json'}")

print("\n" + "=" * 60)
print("HeAR 일반 오디오 테스트 완료!")
print("=" * 60)
print(f"✨ 정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"✨ AUC-ROC: {auc:.4f}")
print(f"📁 결과 디렉토리: {OUTPUT_DIR}")
print("\n⚠️ 데이터 불균형 (1:25)으로 인해 성능이 제한될 수 있습니다.")
print("   Class weighting을 적용하여 완화했습니다.")
