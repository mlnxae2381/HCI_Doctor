"""
저장된 CXR Foundation 임베딩으로 분류 학습
이미 추출된 임베딩을 불러와서 빠르게 평가
"""

import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import json

OUTPUT_DIR = Path("/home/students/cs/doctor/AI_Doctor/test/cxr_results")

print("=" * 60)
print("CXR Foundation - 저장된 임베딩으로 분류 평가")
print("=" * 60)

# 1. 저장된 임베딩 로드
print("\n[1단계] 저장된 임베딩 로드")
X_train = np.load(OUTPUT_DIR / "train_embeddings.npy")
y_train = np.load(OUTPUT_DIR / "train_labels.npy")
X_val = np.load(OUTPUT_DIR / "val_embeddings.npy")
y_val = np.load(OUTPUT_DIR / "val_labels.npy")

print(f"  Train: {X_train.shape}")
print(f"  Val: {X_val.shape}")

# 2. Flatten (4D → 2D)
print("\n[2단계] 임베딩 Flatten")
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)

print(f"  Train: {X_train.shape} → {X_train_flat.shape}")
print(f"  Val: {X_val.shape} → {X_val_flat.shape}")

# 3. 로지스틱 회귀 학습
print("\n[3단계] 로지스틱 회귀 학습 및 평가")
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

# 4. 결과 저장
print("\n[4단계] 결과 저장")

results = {
    "model": "google/cxr-foundation",
    "dataset": "Chest_Xray final_dataset (sampled)",
    "train_samples": int(len(y_train)),
    "val_samples": int(len(y_val)),
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
