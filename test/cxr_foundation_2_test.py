import os

# [안전 장치 1] 터미널 시스템 로그 최소화 (C++ 수준 로그 차단)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# [안전 장치 2] 2번 GPU만 사용 (3번 GPU 충돌 방지)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import tensorflow as tf
import tensorflow_text  # SentencepieceOp 지원을 위해 필수
import numpy as np
import glob
from huggingface_hub import snapshot_download
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def run_cxr_foundation_flow():
    print("\n" + "=" * 60)
    print("[3] CXR Foundation - 가공 데이터셋 특징 추출 및 테스트")
    print("=" * 60)

    # 1. 경로 설정
    DATA_DIR = "/home/students/cs/doctor/AI_Doctor/data/AI_Doctor_Data/Chest_Xray/final_dataset"
    model_id = "google/cxr-foundation"

    try:
        # 2. 모델 다운로드 및 경로 설정 (주신 코드 반영)
        print(f"📦 모델 확인 중: {model_id}")
        model_path = snapshot_download(repo_id=model_id)
        elixr_path = os.path.join(model_path, "elixr-c-v2-pooled")

        # 3. 모델 로드 (TFSMLayer 방식)
        if os.path.exists(elixr_path):
            print(f"✅ ELIXR 모델 경로 사용: {elixr_path}")
            model = tf.keras.layers.TFSMLayer(elixr_path, call_endpoint='serving_default')
        else:
            print(f"✅ 기본 모델 경로 사용: {model_path}")
            model = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
        
        print("🚀 CXR Foundation 로드 완료! (2번 GPU 점유)")

        # 4. 가공된 이미지 로드 함수 (데이터 파이프라인)
        def load_and_preprocess(path, label):
            img = tf.io.read_file(path) # CXR 모델은 원본 바이너리를 내부에서 처리함
            return img, label

        def prepare_dataset(subset_name):
            # Normal=0, Pneumonia=1 로 라벨링
            files = glob.glob(os.path.join(DATA_DIR, subset_name, "*/*.png"))
            labels = [0 if "Normal" in f else 1 for f in files]
            
            ds = tf.data.Dataset.from_tensor_slices((files, labels))
            ds = ds.map(load_and_preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
            return ds, len(files)

        # 5. 특징 추출 실행
        print("\n📊 데이터셋 준비 및 특징 추출 시작...")
        train_ds, n_train = prepare_dataset('train')
        val_ds, n_val = prepare_dataset('val')
        
        def get_embeddings(dataset, name):
            all_embs, all_labels = [], []
            for imgs, lbls in dataset:
                # TFSMLayer의 결과는 딕셔너리 형태이므로 첫 번째 값을 취함
                outputs = model(imgs)
                emb_key = list(outputs.keys())[0]
                all_embs.append(outputs[emb_key].numpy())
                all_labels.append(lbls.numpy())
            return np.concatenate(all_embs), np.concatenate(all_labels)

        X_train, y_train = get_embeddings(train_ds, "Train")
        X_val, y_val = get_embeddings(val_ds, "Val")

        # 6. 간단한 평가 (Linear Probe)
        print("⚖️ 로지스틱 회귀로 성능 검증 중...")
        clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        y_pred = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        print("\n" + "✨" * 20)
        print(f"결과 보고: {n_train}장 학습 / {n_val}장 검증")
        print(f"최종 검증 AUC: {auc:.4f}")
        print("CXR Foundation 테스트 성공!")
        print("✨" * 20)
        return True

    except Exception as e:
        print(f"❌ CXR Foundation 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    run_cxr_foundation_flow()