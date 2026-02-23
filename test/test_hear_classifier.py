"""
HeAR 모델 + 분류기 테스트
- HeAR로 512차원 임베딩 추출 (가중치 고정)
- 간단한 분류기 학습 (정상/비정상)
- Few-shot Learning 방식
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from huggingface_hub import login, snapshot_download
login(token=os.environ.get("HF_TOKEN", ""))

import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# GPU 메모리 동적 할당
def setup_tensorflow_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"TensorFlow GPU 설정 완료: {len(gpus)}개 GPU")
        except RuntimeError as e:
            print(f"GPU 설정 오류: {e}")

setup_tensorflow_gpu()

# 경로 설정
DATA_DIR = Path("/home/students/cs/doctor/AI_Doctor/data/coswara_processed")
OUTPUT_DIR = Path("/home/students/cs/doctor/AI_Doctor/test/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# HeAR 모델 요구사항
TARGET_SR = 16000
SEGMENT_LENGTH = 2.0
SAMPLES_PER_SEGMENT = int(TARGET_SR * SEGMENT_LENGTH)

print("=" * 60)
print("HeAR + 분류기 테스트")
print("=" * 60)

# ============================================
# 1. HeAR 모델 로드
# ============================================
print("\n[1단계] HeAR 모델 로드")

model_id = "google/hear"
print(f"모델 다운로드: {model_id}")
model_path = snapshot_download(repo_id=model_id)
print(f"모델 경로: {model_path}")

# TFSMLayer로 로드
print("HeAR 로딩 중...")
hear_model = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
print("HeAR 로드 완료!")
print("- 입력: (batch, 32000) float32 오디오")
print("- 출력: (batch, 512) 임베딩")

# ============================================
# 2. 데이터 로드
# ============================================
print("\n[2단계] 전처리된 데이터 로드")

metadata_path = DATA_DIR / "metadata.json"
if not metadata_path.exists():
    print(f"오류: {metadata_path} 파일이 없습니다.")
    print("먼저 preprocess_coswara_audio.py를 실행하세요.")
    exit(1)

with open(metadata_path, 'r') as f:
    metadata = json.load(f)

print(f"총 샘플 수: {len(metadata)}")

# 오디오 타입별 분리
audio_type = "cough-heavy"  # 깊은 기침 데이터 사용
filtered_data = [item for item in metadata if item['audio_type'] == audio_type]
print(f"{audio_type} 샘플 수: {len(filtered_data)}")

# 라벨별 개수
normal_count = sum(1 for item in filtered_data if item['label'] == 0)
abnormal_count = sum(1 for item in filtered_data if item['label'] == 1)
print(f"  정상: {normal_count}")
print(f"  비정상: {abnormal_count}")

# ============================================
# 3. HeAR 임베딩 추출
# ============================================
print("\n[3단계] HeAR 임베딩 추출 (3억개 사전학습 활용)")

def load_audio_segment(file_path):
    """오디오 파일 로드"""
    full_path = DATA_DIR / file_path
    # TensorFlow로 WAV 로드
    audio_binary = tf.io.read_file(str(full_path))
    audio, sr = tf.audio.decode_wav(audio_binary, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)

    # 32000 샘플로 맞추기
    if tf.shape(audio)[0] < SAMPLES_PER_SEGMENT:
        padding = SAMPLES_PER_SEGMENT - tf.shape(audio)[0]
        audio = tf.pad(audio, [[0, padding]])
    else:
        audio = audio[:SAMPLES_PER_SEGMENT]

    return audio

# 임베딩 추출
print(f"임베딩 추출 중... (샘플 수: {len(filtered_data)})")

embeddings = []
labels = []
batch_size = 32

for i in range(0, len(filtered_data), batch_size):
    batch_data = filtered_data[i:i+batch_size]
    batch_audios = []

    for item in batch_data:
        try:
            audio = load_audio_segment(item['file_path'])
            batch_audios.append(audio)
            labels.append(item['label'])
        except Exception as e:
            print(f"오디오 로드 실패: {item['file_path']} - {e}")
            continue

    if batch_audios:
        # 배치로 HeAR 실행
        batch_tensor = tf.stack(batch_audios)
        batch_embeddings = hear_model(batch_tensor)

        # 출력 키 확인
        if isinstance(batch_embeddings, dict):
            # 첫 번째 키의 값 사용 (보통 'output_0' 등)
            key = list(batch_embeddings.keys())[0]
            batch_embeddings = batch_embeddings[key]

        embeddings.append(batch_embeddings.numpy())

    if (i // batch_size) % 10 == 0:
        print(f"  진행: {i}/{len(filtered_data)}")

embeddings = np.vstack(embeddings)
labels = np.array(labels)

print(f"임베딩 추출 완료!")
print(f"  임베딩 shape: {embeddings.shape}")
print(f"  라벨 shape: {labels.shape}")

# ============================================
# 4. Train/Val/Test 분할
# ============================================
print("\n[4단계] 데이터 분할")

X_train, X_temp, y_train, y_temp = train_test_split(
    embeddings, labels, test_size=0.3, random_state=42, stratify=labels
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train: {X_train.shape[0]} (정상: {(y_train==0).sum()}, 비정상: {(y_train==1).sum()})")
print(f"Val:   {X_val.shape[0]} (정상: {(y_val==0).sum()}, 비정상: {(y_val==1).sum()})")
print(f"Test:  {X_test.shape[0]} (정상: {(y_test==0).sum()}, 비정상: {(y_test==1).sum()})")

# 클래스 가중치 계산
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(f"\n클래스 가중치: {class_weight_dict}")
print(f"  정상(0): {class_weight_dict[0]:.3f}")
print(f"  비정상(1): {class_weight_dict[1]:.3f}")

# ============================================
# 5. 간단한 분류기 학습
# ============================================
print("\n[5단계] 분류기 학습 (HeAR는 고정, 분류기만 학습)")
print("  - 클래스 가중치 적용: 소수 클래스(비정상)에 더 높은 가중치")

# 모델 구성
classifier = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(512,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
], name='hear_classifier')

classifier.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

classifier.summary()

# 학습
print("\n학습 시작...")
history = classifier.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    class_weight=class_weight_dict,  # 클래스 가중치 적용
    verbose=1
)

# ============================================
# 6. 평가
# ============================================
print("\n[6단계] 테스트 세트 평가")

# 예측
y_pred_proba = classifier.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# 분류 리포트
print("\n분류 리포트:")
print(classification_report(y_test, y_pred, target_names=['정상', '비정상']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['정상', '비정상'],
            yticklabels=['정상', '비정상'])
plt.title('Confusion Matrix - HeAR + Classifier')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"Confusion Matrix 저장: {OUTPUT_DIR / 'confusion_matrix.png'}")

# 학습 곡선
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

# Accuracy
axes[1].plot(history.history['accuracy'], label='Train Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[1].set_title('Training and Validation Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'training_curves.png', dpi=300, bbox_inches='tight')
print(f"학습 곡선 저장: {OUTPUT_DIR / 'training_curves.png'}")

# ============================================
# 7. 모델 저장
# ============================================
print("\n[7단계] 모델 저장")

classifier.save(OUTPUT_DIR / 'hear_classifier.h5')
print(f"분류기 저장: {OUTPUT_DIR / 'hear_classifier.h5'}")

# 결과 요약 저장
results = {
    "audio_type": audio_type,
    "total_samples": len(filtered_data),
    "train_samples": int(X_train.shape[0]),
    "val_samples": int(X_val.shape[0]),
    "test_samples": int(X_test.shape[0]),
    "test_accuracy": float(np.mean(y_pred == y_test)),
    "confusion_matrix": cm.tolist()
}

with open(OUTPUT_DIR / 'results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"결과 저장: {OUTPUT_DIR / 'results.json'}")

print("\n" + "=" * 60)
print("테스트 완료!")
print("=" * 60)
print(f"테스트 정확도: {results['test_accuracy']:.4f}")
print(f"결과 디렉토리: {OUTPUT_DIR}")
