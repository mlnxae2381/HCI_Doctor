import os
import sys
import glob
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from huggingface_hub import login, snapshot_download
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# [안전 장치 1] 터미널에 불필요한 시스템 로그(C++ 수준) 출력 금지
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

def run_final_process():
    # --- 설정 영역 ---
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    DATA_DIR = "/home/students/cs/doctor/AI_Doctor/data/AI_Doctor_Data/Chest_Xray/final_dataset"
    SAVE_DIR = "/home/students/cs/doctor/AI_Doctor/test/cxr_results"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # [안전 장치 2] 데이터 경로 및 파일 존재 유무 선제적 확인
    train_files = glob.glob(os.path.join(DATA_DIR, "train/*/*.png"))
    val_files = glob.glob(os.path.join(DATA_DIR, "val/*/*.png"))
    
    print(f"📊 [데이터 확인] Train: {len(train_files)}장 / Val: {len(val_files)}장")
    if len(train_files) == 0:
        print("❌ 에러: 데이터를 찾을 수 없습니다. 경로를 다시 확인하세요!")
        return

    # 1. 모델 로드
    ##login(token=HF_TOKEN)
    #model_id = "google/cxr-foundation"
    #model_path = snapshot_download(repo_id=model_id)
    #print("🤖 모델 로딩 중 (serving_default)...")
    #model = hub.KerasLayer(model_path, signature="serving_default", signature_outputs_as_dict=True)
    # 1. 모델 로드 (Kaggle Models 통합 주소 사용)
    print("🤖 모델 로딩 중 (Kaggle/TF-Hub 최신 주소)...")
    
    # TF-Hub가 Kaggle로 통합되면서 사용되는 가장 확실한 경로입니다.
    model_url = "https://www.kaggle.com/models/google/cxr-foundation/frameworks/TensorFlow2/variations/cxr-foundation/versions/1"
    
    try:
        model = hub.KerasLayer(model_url, signature="serving_default", signature_outputs_as_dict=True)
        print("✅ 드디어 모델 로드 성공!")
    except Exception as e:
        print(f"❌ Kaggle 주소 로드 실패: {e}")
        print("🆘 모든 리모트 주소가 실패했습니다. 로컬 파일을 강제로 열어보겠습니다.")
        
        # [마지막 수단] 아까 HF로 받았던 폴더 경로를 다시 사용 (변수 수동 입력)
        # 아까 에러 메시지에 떴던 그 긴 경로를 복사해서 넣으세요.
        local_path = "/home/students/cs/doctor/.cache/huggingface/hub/models--google--cxr-foundation/snapshots/e5af8ea44a17bad5504f7e485388d6b05786860f"
        try:
            model = hub.KerasLayer(local_path, signature="serving_default", signature_outputs_as_dict=True)
            print("✅ 로컬 스냅샷으로 로드 성공!")
        except Exception as e3:
            print(f"❌ 최종 실패: {e3}")
            return
    
    # 2. 데이터셋 로더 (0~1 float32)
    def load_ds(subset):
        return tf.keras.utils.image_dataset_from_directory(
            os.path.join(DATA_DIR, subset),
            image_size=(224, 224),
            batch_size=128,
            shuffle=False
        ).map(lambda x, y: (x / 255.0, y)).prefetch(tf.data.AUTOTUNE)

    train_ds = load_ds('train')
    val_ds = load_ds('val')

    # 3. 특징 추출 함수 (바이너리 출력 원천 차단)
    def extract(dataset, name):
        print(f"🚀 {name} 특징 추출 시작 (Batch 단위 인코딩)...")
        embs, lbls = [], []
        
        for i, (imgs, labels) in enumerate(dataset):
            # 픽셀 데이터를 JPEG 문자열로 변환 (절대 출력하지 않음)
            input_strings = tf.map_fn(
                lambda x: tf.io.encode_jpeg(tf.cast(x * 255, tf.uint8)), 
                imgs, 
                fn_output_signature=tf.string
            )
            
            # 모델 추론
            outputs = model(input_strings)
            embedding = list(outputs.values())[0] 
            
            embs.append(embedding.numpy())
            lbls.append(labels.numpy())
            
            # [안전 장치 3] 진행 상황만 숫자로 표시 (10배치마다)
            if (i+1) % 10 == 0:
                print(f"   > {name} 진행도: {(i+1)*128}장 완료...")

        X = np.concatenate(embs)
        y = np.concatenate(lbls).flatten()
        
        # 중간 저장 (혹시 모를 중단 대비)
        np.save(os.path.join(SAVE_DIR, f"{name}_X.npy"), X)
        np.save(os.path.join(SAVE_DIR, f"{name}_y.npy"), y)
        return X, y

    # 4. 전체 실행
    X_train, y_train = extract(train_ds, "Train")
    X_val, y_val = extract(val_ds, "Validation")

    # 5. 최종 분석
    print("📊 로지스틱 회귀 모델 학습 및 평가 중...")
    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    probs = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, probs)

    print("\n" + "="*50)
    print(f"🏆 최종 실험 결과 보고")
    print(f"   - AUC-ROC 점수: {auc:.4f}")
    print(f"   - 결과 파일 저장 완료: {SAVE_DIR}")
    print("="*50)

if __name__ == "__main__":
    run_final_process()