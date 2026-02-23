import os
import tensorflow as tf
import numpy as np
from huggingface_hub import snapshot_download

# GPU 2번 고정
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def test_single_file():
    # 1. 모델 로드
    model_id = "google/hear"
    model_path = snapshot_download(repo_id=model_id)
    model = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
    print("✅ 모델 로드 성공")

    # 2. 실제 데이터 하나 지정 (본인의 파일 경로로 수정하세요!)
    test_file = "/home/students/cs/doctor/AI_Doctor/data/AI_Doctor_Data/classified_audio/Normal/102_1b1_Ar_sc_Meditron.wav"
    
    if not os.path.exists(test_file):
        print(f"❌ 파일을 찾을 수 없습니다: {test_file}")
        return

    # 3. 전처리 (상자 크기 맞추기: 16kHz, 2초 = 32000 samples)
    raw_audio = tf.io.read_file(test_file)
    audio, _ = tf.audio.decode_wav(raw_audio, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)

    # 길이 조절 로직
    if tf.shape(audio)[0] < 32000:
        audio = tf.pad(audio, [[0, 32000 - tf.shape(audio)[0]]])
    else:
        audio = audio[:32000]
    
    input_tensor = tf.expand_dims(audio, axis=0) # (1, 32000)

    # 4. 추론 테스트
    output = model(input_tensor)
    print(f"📍 테스트 결과 성공! 임베딩 shape: {output['output_0'].shape}")

if __name__ == "__main__":
    test_single_file()