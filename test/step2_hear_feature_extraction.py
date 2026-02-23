# 파일명: step2_hear_feature_extraction.py
# 설명: HeAR 모델을 사용하여 호흡음 오디오 파일에서 특징 추출을 수행합니다.
# 작성일: 2026-02-10
import os
import numpy as np
import tensorflow as tf
from huggingface_hub import snapshot_download
from tqdm import tqdm
import glob  # 하위 폴더 검색을 위해 필요
import librosa

# 1. GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

def run_extraction():
    # 경로 설정
    DATA_DIR = "/home/students/cs/doctor/AI_Doctor/data/AI_Doctor_Data/classified_audio"
    SAVE_DIR = "/home/students/cs/doctor/AI_Doctor/results"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 2. 모델 로드
    print("📡 HeAR 모델 로딩 중...")
    model_path = snapshot_download(repo_id="google/hear")
    model = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')

    # 3. 하위 폴더를 포함하여 모든 .wav 파일 찾기 (** 사용)
    # recursive=True를 설정해야 하위 폴더까지 들어갑니다.
    print(f"📂 데이터 검색 중: {DATA_DIR}")
    audio_files = glob.glob(os.path.join(DATA_DIR, "**/*.wav"), recursive=True)
    
    print(f"🚀 총 {len(audio_files)}개 파일 발견!")

    if len(audio_files) == 0:
        print("❌ 파일을 찾지 못했습니다. 경로와 확장자를 확인해주세요.")
        return

    all_embeddings = []
    file_list = []

    # 4. 특징 추출 루프
    
    for f_path in tqdm(audio_files, desc="Extracting Features"):
        try:
            f_name = os.path.basename(f_path)
            
            # --- 수정된 로드 방식 ---
            # librosa는 24비트를 자동으로 처리해줍니다.
            # sr=16000 또는 32000 등 HeAR 모델이 원하는 샘플링 레이트로 로드
            audio, _ = librosa.load(f_path, sr=16000) 
            
            # 모델 입력 규격(2초, 32000 samples) 맞춤
            if len(audio) < 32000:
                audio = np.pad(audio, (0, 32000 - len(audio)))
            else:
                audio = audio[:32000]
            
            # TensorFlow 텐서로 변환
            audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
            
            output = model(tf.expand_dims(audio_tensor, axis=0))
            # -----------------------
            
            all_embeddings.append(output['output_0'].numpy())
            file_list.append(f_path) 
            
        except Exception as e:
            print(f"⚠️ {f_path} 에러 발생: {e}")

    # 5. 결과 저장
    save_path = os.path.join(SAVE_DIR, "breath_embeddings.npy")
    np.save(save_path, {
        "embeddings": np.vstack(all_embeddings), 
        "filenames": file_list
    })
    print(f"✅ 추출 완료! 저장 위치: {save_path}")

if __name__ == "__main__":
    run_extraction()