"""
CXR Foundation 테스트 스크립트 (Clean Version)
Chest X-ray 데이터셋으로 임베딩 추출 및 분류 평가
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU 1번 사용
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # TensorFlow 로그 최소화
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow_text  # SentencepieceOp 필수
import numpy as np

# NumPy 출력 억제
np.set_printoptions(threshold=10, edgeitems=2)

# TensorFlow 로깅 완전 억제
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
from pathlib import Path
from huggingface_hub import snapshot_download, login
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from dotenv import load_dotenv
import json

# .env 로드
load_dotenv()

# 경로 설정
DATA_DIR = Path("/home/students/cs/doctor/AI_Doctor/data/AI_Doctor_Data/Chest_Xray/final_dataset")
OUTPUT_DIR = Path("/home/students/cs/doctor/AI_Doctor/test/cxr_results")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("CXR Foundation - Chest X-ray 분류 테스트")
print("=" * 60)

# ============================================
# 1. 모델 로드
# ============================================
print("\n[1단계] CXR Foundation 모델 로드")

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

model_id = "google/cxr-foundation"
print(f"모델 다운로드 중: {model_id}")

model_path = snapshot_download(repo_id=model_id)
elixr_path = os.path.join(model_path, "elixr-c-v2-pooled")

if os.path.exists(elixr_path):
    print(f"ELIXR 모델 경로 사용: {elixr_path}")
    model = tf.keras.layers.TFSMLayer(elixr_path, call_endpoint='serving_default')
else:
    print(f"기본 모델 경로 사용: {model_path}")
    model = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')

print("✅ CXR Foundation 로드 완료!")

# ============================================
# 2. 데이터셋 준비
# ============================================
print("\n[2단계] 데이터셋 로드")

def load_image_paths(subset):
    """이미지 경로와 레이블 로드"""
    normal_dir = DATA_DIR / subset / "Normal"
    abnormal_dir = DATA_DIR / subset / "Abnormal"

    normal_files = list(normal_dir.glob("*.png"))
    abnormal_files = list(abnormal_dir.glob("*.png"))

    paths = [str(p) for p in normal_files + abnormal_files]
    labels = [0] * len(normal_files) + [1] * len(abnormal_files)

    return paths, labels

train_paths, train_labels = load_image_paths("train")
val_paths, val_labels = load_image_paths("val")

print(f"Train: {len(train_paths)}장 (Normal: {train_labels.count(0)}, Abnormal: {train_labels.count(1)})")
print(f"Val: {len(val_paths)}장 (Normal: {val_labels.count(0)}, Abnormal: {val_labels.count(1)})")

# ============================================
# 3. 임베딩 추출 함수
# ============================================
print("\n[3단계] 임베딩 추출")

def extract_embeddings(image_paths, labels, name="Dataset"):
    """
    CXR Foundation으로 이미지 임베딩 추출
    CXR Foundation은 tf.Example 형식으로 한 번에 하나씩 처리
    """
    embeddings = []

    print(f"{name} 임베딩 추출 중... (총 {len(image_paths)}장)")

    for i, path in enumerate(image_paths):
        # 이미지 읽기
        img = tf.io.read_file(path)
        # PNG로 디코딩
        img_decoded = tf.image.decode_png(img, channels=1)
        # 224x224로 리사이즈
        img_resized = tf.image.resize(img_decoded, [224, 224])
        # uint8로 변환 후 PNG 인코딩
        img_uint8 = tf.cast(img_resized, tf.uint8)
        img_encoded = tf.io.encode_png(img_uint8)

        # tf.Example 형식으로 변환
        feature = {
            'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_encoded.numpy()])),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        serialized_example = example.SerializeToString()

        # string tensor로 변환 (배치 크기 1)
        input_tensor = tf.constant([serialized_example], dtype=tf.string)

        # 모델 추론
        outputs = model(input_tensor)

        # 첫 번째 출력 키 사용
        emb_key = list(outputs.keys())[0]
        embedding = outputs[emb_key].numpy()

        embeddings.append(embedding)

        # 진행 상황 출력 (100장마다)
        if (i + 1) % 100 == 0:
            print(f"  진행: {i + 1}/{len(image_paths)}장")

    return np.concatenate(embeddings, axis=0)

# 샘플링 (한 번에 하나씩 처리하므로 샘플 크기 줄임)
import random
random.seed(42)

TRAIN_SAMPLE_SIZE = 2000  # Train 샘플링
VAL_SAMPLE_SIZE = 1000    # Val 샘플링

print(f"\n⚠️ 데이터 샘플링 (한 번에 하나씩 처리)")
print(f"   Train: {len(train_paths)}장 → {TRAIN_SAMPLE_SIZE}장")
print(f"   Val: {len(val_paths)}장 → {VAL_SAMPLE_SIZE}장")

# Train 샘플링
if len(train_paths) > TRAIN_SAMPLE_SIZE:
    indices = random.sample(range(len(train_paths)), TRAIN_SAMPLE_SIZE)
    train_paths_sampled = [train_paths[i] for i in indices]
    train_labels_sampled = [train_labels[i] for i in indices]
else:
    train_paths_sampled = train_paths
    train_labels_sampled = train_labels

# Val 샘플링
if len(val_paths) > VAL_SAMPLE_SIZE:
    indices = random.sample(range(len(val_paths)), VAL_SAMPLE_SIZE)
    val_paths_sampled = [val_paths[i] for i in indices]
    val_labels_sampled = [val_labels[i] for i in indices]
else:
    val_paths_sampled = val_paths
    val_labels_sampled = val_labels

print("\nTrain 임베딩 추출 시작...")
X_train = extract_embeddings(train_paths_sampled, train_labels_sampled, name="Train")
y_train = np.array(train_labels_sampled)

print("\nVal 임베딩 추출 시작...")
X_val = extract_embeddings(val_paths_sampled, val_labels_sampled, name="Val")
y_val = np.array(val_labels_sampled)

print(f"\n임베딩 추출 완료!")
print(f"  Train: {X_train.shape}")
print(f"  Val: {X_val.shape}")

# 4차원 임베딩을 2차원으로 flatten (LogisticRegression을 위해)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)

print(f"\n임베딩 Flatten:")
print(f"  Train: {X_train.shape} → {X_train_flat.shape}")
print(f"  Val: {X_val.shape} → {X_val_flat.shape}")

# 임베딩 저장 (원본 + flatten 모두 저장)
np.save(OUTPUT_DIR / "train_embeddings.npy", X_train)
np.save(OUTPUT_DIR / "train_embeddings_flat.npy", X_train_flat)
np.save(OUTPUT_DIR / "train_labels.npy", y_train)
np.save(OUTPUT_DIR / "val_embeddings.npy", X_val)
np.save(OUTPUT_DIR / "val_embeddings_flat.npy", X_val_flat)
np.save(OUTPUT_DIR / "val_labels.npy", y_val)
print(f"\n임베딩 저장: {OUTPUT_DIR}")

# ============================================
# 4. 분류 모델 학습 및 평가
# ============================================
print("\n[4단계] 로지스틱 회귀 학습 및 평가")

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_flat, y_train)

# 예측
y_pred = clf.predict(X_val_flat)
y_pred_proba = clf.predict_proba(X_val_flat)[:, 1]

# 평가
accuracy = accuracy_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_pred_proba)

print(f"\n✨ 평가 결과:")
print(f"  정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  AUC-ROC: {auc:.4f}")

print("\n분류 리포트:")
print(classification_report(y_val, y_pred, target_names=["Normal", "Abnormal"]))

# ============================================
# 5. 결과 저장
# ============================================
print("\n[5단계] 결과 저장")

results = {
    "model": "google/cxr-foundation",
    "dataset": "Chest_Xray final_dataset (sampled)",
    "train_samples": len(train_paths_sampled),
    "val_samples": len(val_paths_sampled),
    "accuracy": float(accuracy),
    "auc_roc": float(auc),
    "classification_report": classification_report(y_val, y_pred, target_names=["Normal", "Abnormal"], output_dict=True)
}

with open(OUTPUT_DIR / "cxr_test_results.json", 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"결과 저장: {OUTPUT_DIR / 'cxr_test_results.json'}")

print("\n" + "=" * 60)
print("CXR Foundation 테스트 완료!")
print("=" * 60)
print(f"✨ 정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"✨ AUC-ROC: {auc:.4f}")
print(f"📁 결과 디렉토리: {OUTPUT_DIR}")
