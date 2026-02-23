import os

# [안전 장치] 터미널 로그 억제 및 2번 GPU 고정
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import tensorflow as tf
import numpy as np
from huggingface_hub import login, snapshot_download

def test_hear():
    print("\n" + "=" * 60)
    print("[2] HeAR - 건강 음향 분석 (기침, 호흡음)")
    print("=" * 60)

    try:
        # 1. Hugging Face 로그인 (필요시 토큰 입력)
        login(token=os.environ.get("HF_TOKEN", "")) 
        
        model_id = "google/hear"
        print(f"📡 모델 다운로드 시도: {model_id}")

        # 2. 모델 다운로드 및 경로 확인
        model_path = snapshot_download(repo_id=model_id)
        print(f"📂 로컬 모델 경로: {model_path}")

        # 3. TFSMLayer로 로드 (2번 GPU 사용)
        print("🤖 TFSMLayer 로딩 중 (Device: GPU 2)...")
        # HeAR 모델은 보통 오디오 파형을 직접 입력받으므로 TFSMLayer가 적합합니다.
        model = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')

        print("✅ HeAR 로드 완료!")
        print("\n📝 모델 정보:")
        print("- 입력: 2초 오디오 파형 (16kHz Mono, shape: (batch, 32000))")
        print("- 출력: 고차원 특징 임베딩 (Acoustic Embeddings)")

        # 4. 더미 오디오로 테스트
        # 2개 샘플, 각 32,000 샘플(2초) 생성
        print("\n🧪 더미 데이터 추론 테스트 시작...")
        dummy_audio = tf.random.normal(shape=(2, 32000), dtype=tf.float32)
        
        # 모델 추론
        output = model(dummy_audio)

        print(f"📍 입력 shape: {dummy_audio.shape}")
        print(f"📍 출력 keys: {list(output.keys())}")
        
        # 각 출력 노드의 결과값 출력
        for key, value in output.items():
            print(f"   > {key}: shape={value.shape}")

        print("\n🎉 HeAR 테스트 최종 성공!")
        return True

    except Exception as e:
        print(f"❌ HeAR 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # GPU 메모리 동적 할당 설정 (필요시)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            
    test_hear()