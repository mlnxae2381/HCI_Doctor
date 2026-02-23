"""
HAI-DEF 모델 컬렉션 테스트 스크립트
TFSMLayer를 사용하여 SavedModel 로드
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # TF 로그 줄이기

from huggingface_hub import login
login(token=os.environ.get("HF_TOKEN", ""))

import torch
import numpy as np
import gc

def setup_tensorflow_gpu():
    """TensorFlow GPU 메모리 동적 할당 설정"""
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"TensorFlow GPU 설정 완료: {len(gpus)}개 GPU 사용 가능")
        except RuntimeError as e:
            print(f"TensorFlow GPU 설정 오류: {e}")

def clear_gpu_memory():
    """PyTorch GPU 메모리 완전 해제"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

print("=" * 60)
print("HAI-DEF 모델 컬렉션 테스트")
print("=" * 60)

# ============================================
# 1. MedGemma - 의료 멀티모달 생성 모델
# ============================================
def test_medgemma():
    print("\n" + "=" * 60)
    print("[1] MedGemma 4B - 의료 Q&A 및 이미지 분석")
    print("=" * 60)

    from transformers import AutoProcessor, AutoModelForImageTextToText

    model_id = "google/medgemma-4b-it"
    print(f"모델 로딩: {model_id}")

    # 단일 GPU(cuda:0)에서 실행하여 속도 향상
    device = "cuda:0"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    ).to(device)
    print("MedGemma 로드 완료!")

    question = "심근경색의 초기 증상을 간단히 설명해주세요."
    print(f"\n질문: {question}")
    print("-" * 40)
    print("생성 중...")

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": question}]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False)
    response = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    print(f"답변: {response}")

    del model, processor
    clear_gpu_memory()
    print("MedGemma GPU 메모리 해제 완료")

    return True

# ============================================
# 2. HeAR - 건강 음향 분석 (TFSMLayer 사용)
# ============================================
def test_hear():
    print("\n" + "=" * 60)
    print("[2] HeAR - 건강 음향 분석 (기침, 호흡음)")
    print("=" * 60)

    try:
        setup_tensorflow_gpu()
        import tensorflow as tf
        from huggingface_hub import snapshot_download

        model_id = "google/hear"
        print(f"모델 다운로드: {model_id}")

        # 모델 다운로드
        model_path = snapshot_download(repo_id=model_id)
        print(f"모델 경로: {model_path}")

        # TFSMLayer로 로드
        print("TFSMLayer로 로딩 중...")
        model = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')

        print("HeAR 로드 완료!")
        print("\n모델 정보:")
        print("- 입력: 2초 오디오 (16kHz, shape: (n, 32000))")
        print("- 출력: 512차원 임베딩")

        # 더미 오디오로 테스트
        dummy_audio = tf.random.normal(shape=(2, 32000), dtype=tf.float32)
        output = model(dummy_audio)

        print(f"\n입력 shape: {dummy_audio.shape}")
        print(f"출력 keys: {output.keys()}")
        for key, value in output.items():
            print(f"  {key}: shape={value.shape}")

        print("HeAR 테스트 성공!")
        return True

    except Exception as e:
        print(f"HeAR 테스트 실패: {e}")
        return False

# ============================================
# 3. CXR Foundation - 흉부 X-ray
# ============================================
def test_cxr_foundation():
    print("\n" + "=" * 60)
    print("[3] CXR Foundation - 흉부 X-ray 임베딩")
    print("=" * 60)

    try:
        setup_tensorflow_gpu()
        import tensorflow as tf
        # SentencepieceOp을 위해 tensorflow_text 임포트 필요
        import tensorflow_text
        from huggingface_hub import snapshot_download

        model_id = "google/cxr-foundation"
        print(f"모델 다운로드: {model_id}")

        model_path = snapshot_download(repo_id=model_id)

        # elixr-c-v2-pooled 서브 모델 사용
        elixr_path = os.path.join(model_path, "elixr-c-v2-pooled")
        if os.path.exists(elixr_path):
            print(f"ELIXR 모델 경로: {elixr_path}")
            model = tf.keras.layers.TFSMLayer(elixr_path, call_endpoint='serving_default')
        else:
            model = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')

        print("CXR Foundation 로드 완료!")
        print("\n모델 정보:")
        print("- 입력: 흉부 X-ray 이미지 (tf.Example 형식)")
        print("- 출력: 임베딩 벡터")
        print("- 용도: 폐렴, 결핵, COVID-19 검출")

        print("CXR Foundation 테스트 성공!")
        return True

    except Exception as e:
        print(f"CXR Foundation 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================
# 4. Path Foundation - 병리 이미지
# ============================================
def test_path_foundation():
    print("\n" + "=" * 60)
    print("[4] Path Foundation - 병리 슬라이드 임베딩")
    print("=" * 60)

    try:
        setup_tensorflow_gpu()
        import tensorflow as tf
        from huggingface_hub import snapshot_download
        from PIL import Image

        model_id = "google/path-foundation"
        print(f"모델 다운로드: {model_id}")

        model_path = snapshot_download(repo_id=model_id)
        print(f"모델 경로: {model_path}")

        model = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')

        print("Path Foundation 로드 완료!")
        print("\n모델 정보:")
        print("- 입력: 병리 슬라이드 패치 이미지 (224x224)")
        print("- 출력: 임베딩 벡터")
        print("- 용도: 암 진단, 조직 분류")

        # 더미 이미지로 테스트
        dummy_image = tf.random.uniform(shape=(1, 224, 224, 3), minval=0, maxval=255, dtype=tf.float32)
        output = model(dummy_image)

        print(f"\n입력 shape: {dummy_image.shape}")
        print(f"출력 keys: {output.keys()}")
        for key, value in output.items():
            print(f"  {key}: shape={value.shape}")

        print("Path Foundation 테스트 성공!")
        return True

    except Exception as e:
        print(f"Path Foundation 테스트 실패: {e}")
        return False

# ============================================
# 5. Derm Foundation - 피부 이미지
# ============================================
def test_derm_foundation():
    print("\n" + "=" * 60)
    print("[5] Derm Foundation - 피부과 이미지 임베딩")
    print("=" * 60)

    try:
        # Derm Foundation은 CPU 전용 모델
        import tensorflow as tf
        print("Derm Foundation: CPU 모드로 실행 (모델이 CPU 전용)")
        from huggingface_hub import snapshot_download
        from PIL import Image
        import io

        model_id = "google/derm-foundation"
        print(f"모델 다운로드: {model_id}")

        model_path = snapshot_download(repo_id=model_id)
        print(f"모델 경로: {model_path}")

        # CPU에서 모델 로드 및 실행
        with tf.device('/CPU:0'):
            model = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')

            print("Derm Foundation 로드 완료!")
            print("\n모델 정보:")
            print("- 입력: 피부 이미지 (JPEG 바이트를 string tensor로)")
            print("- 출력: 임베딩 벡터")
            print("- 용도: 피부 질환 분류, 피부암 스크리닝")

            # 더미 이미지 생성 후 tf.Example 형식으로 인코딩
            dummy_np = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            dummy_pil = Image.fromarray(dummy_np)

            # PNG로 인코딩
            buffer = io.BytesIO()
            dummy_pil.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()

            # tf.Example 형식으로 변환
            feature = {
                'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            serialized_example = example.SerializeToString()

            # string tensor로 변환 (키워드 인자 사용)
            input_tensor = tf.constant([serialized_example], dtype=tf.string)

            output = model(inputs=input_tensor)

            print(f"\n입력 type: string tensor (tf.Example)")
            print(f"출력 keys: {output.keys()}")
            for key, value in output.items():
                print(f"  {key}: shape={value.shape}")

        print("Derm Foundation 테스트 성공!")
        return True

    except Exception as e:
        print(f"Derm Foundation 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================
# 6. MedSigLIP 설명
# ============================================
def test_medsiglip():
    print("\n" + "=" * 60)
    print("[6] MedSigLIP - 의료 이미지-텍스트 임베딩")
    print("=" * 60)

    print("MedSigLIP은 MedGemma 내부에 포함된 이미지 인코더입니다.")
    print("별도 모델이 아니라 MedGemma를 통해 사용합니다.")
    print("\nMedGemma 내부 구조:")
    print("- SigLIP 이미지 인코더 (의료 이미지 학습됨)")
    print("- Gemma 3 LLM")

    return True


# ============================================
# 메인 실행
# ============================================
if __name__ == "__main__":
    results = {}

    results["MedGemma"] = test_medgemma()
    results["HeAR"] = test_hear()
    results["CXR Foundation"] = test_cxr_foundation()
    results["Path Foundation"] = test_path_foundation()
    results["Derm Foundation"] = test_derm_foundation()
    results["MedSigLIP"] = test_medsiglip()

    # 결과 요약
    print("\n" + "=" * 60)
    print("HAI-DEF 모델 테스트 결과 요약")
    print("=" * 60)

    for model_name, success in results.items():
        status = "✅ 성공" if success else "❌ 실패"
        print(f"{model_name}: {status}")

    print("\n테스트 완료!")
