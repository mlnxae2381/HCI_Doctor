"""
Coswara 오디오 데이터 전처리 스크립트
- 기침/호흡 데이터 추출
- 16kHz 리샘플링
- 2초 분할 (HeAR 모델 요구사항)
- 품질 필터링 (excellent만 사용)
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import librosa
import soundfile as sf
from tqdm import tqdm

# 경로 설정
BASE_DIR = Path("/home/students/cs/doctor/AI_Doctor/data/Coswara-Data-master")
EXTRACTED_DIR = BASE_DIR / "Extracted_data"
OUTPUT_DIR = Path("/home/students/cs/doctor/AI_Doctor/data/coswara_processed")
ANNOTATIONS_DIR = BASE_DIR / "annotations"

# HeAR 모델 요구사항
TARGET_SR = 16000  # 16kHz
SEGMENT_LENGTH = 2.0  # 2초
SAMPLES_PER_SEGMENT = int(TARGET_SR * SEGMENT_LENGTH)  # 32,000 샘플

# 사용할 오디오 타입
AUDIO_TYPES = ["cough-heavy", "cough-shallow", "breathing-deep", "breathing-shallow"]

def load_metadata():
    """메타데이터 로드"""
    csv_path = BASE_DIR / "combined_data.csv"
    df = pd.read_csv(csv_path)
    print(f"총 참가자 수: {len(df)}")
    return df

def load_quality_labels():
    """품질 라벨 로드"""
    quality_labels = {}

    for audio_type in AUDIO_TYPES:
        label_file = ANNOTATIONS_DIR / f"{audio_type}_labels.csv"
        if label_file.exists():
            df = pd.read_csv(label_file)
            # FILENAME, QUALITY
            quality_dict = dict(zip(df['FILENAME'], df[' QUALITY']))
            quality_labels[audio_type] = quality_dict
            print(f"{audio_type}: {len(quality_dict)} 라벨")

    return quality_labels

def get_label_from_covid_status(covid_status):
    """COVID 상태 -> 정상/비정상 라벨 변환"""
    # 정상
    normal_statuses = ['healthy', 'recovered_full']
    # 비정상 (COVID 양성)
    abnormal_statuses = ['positive_mild', 'positive_moderate', 'positive_asymp']

    if covid_status in normal_statuses:
        return 0  # 정상
    elif covid_status in abnormal_statuses:
        return 1  # 비정상
    else:
        return -1  # 불확실 (제외)

def process_audio_file(audio_path, target_sr=16000):
    """오디오 파일 로드 및 리샘플링"""
    try:
        # 오디오 로드
        audio, sr = librosa.load(audio_path, sr=None)

        # 리샘플링
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        return audio, target_sr
    except Exception as e:
        print(f"오디오 로드 실패 {audio_path}: {e}")
        return None, None

def split_into_segments(audio, segment_samples):
    """오디오를 2초 세그먼트로 분할"""
    segments = []

    # 오디오가 너무 짧으면 패딩
    if len(audio) < segment_samples:
        padded = np.zeros(segment_samples)
        padded[:len(audio)] = audio
        segments.append(padded)
    else:
        # 2초씩 분할
        num_segments = len(audio) // segment_samples
        for i in range(num_segments):
            start = i * segment_samples
            end = start + segment_samples
            segments.append(audio[start:end])

    return segments

def preprocess_coswara_data():
    """Coswara 데이터 전처리 메인 함수"""

    # 메타데이터 로드
    metadata = load_metadata()
    quality_labels = load_quality_labels()

    # 출력 디렉토리 생성
    for audio_type in AUDIO_TYPES:
        for label in ['normal', 'abnormal']:
            output_path = OUTPUT_DIR / audio_type / label
            output_path.mkdir(parents=True, exist_ok=True)

    # 처리 통계
    stats = {
        audio_type: {'normal': 0, 'abnormal': 0, 'excluded': 0}
        for audio_type in AUDIO_TYPES
    }

    # JSON 메타데이터
    processed_data = []

    # 각 참가자별 처리
    print("\n오디오 파일 처리 중...")
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
        user_id = row['id']
        covid_status = row['covid_status']
        label = get_label_from_covid_status(covid_status)

        # 불확실한 상태는 제외
        if label == -1:
            continue

        label_name = 'normal' if label == 0 else 'abnormal'

        # 각 오디오 타입별 처리
        for audio_type in AUDIO_TYPES:
            # 품질 확인 (excellent만 사용)
            audio_filename = f"{user_id}_{audio_type}"
            quality = quality_labels.get(audio_type, {}).get(audio_filename, 0)

            if quality != 2:  # 2 = excellent
                stats[audio_type]['excluded'] += 1
                continue

            # 오디오 파일 찾기 (폴더 구조: Extracted_data/202*/user_id/audio-type.wav)
            audio_path = None
            for date_folder in EXTRACTED_DIR.glob("202*"):
                # user_id 폴더 안에서 파일 찾기
                user_folder = date_folder / user_id
                if user_folder.exists():
                    potential_path = user_folder / f"{audio_type}.wav"
                    if potential_path.exists():
                        audio_path = potential_path
                        break

            if audio_path is None:
                stats[audio_type]['excluded'] += 1
                continue

            # 오디오 처리
            audio, sr = process_audio_file(audio_path)
            if audio is None:
                stats[audio_type]['excluded'] += 1
                continue

            # 2초 세그먼트로 분할
            segments = split_into_segments(audio, SAMPLES_PER_SEGMENT)

            # 각 세그먼트 저장
            for seg_idx, segment in enumerate(segments):
                output_filename = f"{user_id}_{audio_type}_seg{seg_idx}.wav"
                output_path = OUTPUT_DIR / audio_type / label_name / output_filename

                # WAV 파일 저장
                sf.write(output_path, segment, TARGET_SR)

                # 메타데이터 저장
                processed_data.append({
                    "user_id": user_id,
                    "audio_type": audio_type,
                    "segment_idx": seg_idx,
                    "label": label,
                    "label_name": label_name,
                    "covid_status": covid_status,
                    "file_path": str(output_path.relative_to(OUTPUT_DIR)),
                    "duration": SEGMENT_LENGTH,
                    "sample_rate": TARGET_SR
                })

            stats[audio_type][label_name] += len(segments)

    # JSON 메타데이터 저장
    metadata_path = OUTPUT_DIR / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    # 통계 출력
    print("\n" + "=" * 60)
    print("전처리 완료 통계")
    print("=" * 60)

    for audio_type in AUDIO_TYPES:
        total = stats[audio_type]['normal'] + stats[audio_type]['abnormal']
        print(f"\n{audio_type}:")
        print(f"  정상: {stats[audio_type]['normal']} 세그먼트")
        print(f"  비정상: {stats[audio_type]['abnormal']} 세그먼트")
        print(f"  제외됨: {stats[audio_type]['excluded']} 파일")
        print(f"  총합: {total} 세그먼트")

    print(f"\n출력 디렉토리: {OUTPUT_DIR}")
    print(f"메타데이터: {metadata_path}")
    print(f"총 처리 세그먼트: {len(processed_data)}")

if __name__ == "__main__":
    # librosa, soundfile 설치 확인
    try:
        import librosa
        import soundfile as sf
        print("필요 라이브러리 확인 완료")
    except ImportError as e:
        print(f"필요 라이브러리 설치 필요: {e}")
        print("conda activate doctor")
        print("pip install librosa soundfile")
        exit(1)

    # Extracted_data 폴더 확인
    if not EXTRACTED_DIR.exists():
        print(f"경고: {EXTRACTED_DIR} 폴더가 없습니다.")
        print("extract_data.py를 먼저 실행하세요.")
        exit(1)

    preprocess_coswara_data()
