# HeAR 모델 테스트 가이드

## 개요
HeAR (Health Acoustic Representations) 모델과 Coswara 데이터셋을 사용하여 기침/호흡 소리로 정상/비정상을 판단하는 테스트입니다.

## HeAR 모델 특징
- **사전학습 데이터**: 3억 개 오디오 클립 (YouTube 건강 영상)
- **입력**: 2초, 16kHz 오디오
- **출력**: 512차원 임베딩
- **활용**: 파인튜닝 없이 임베딩 추출 → 분류기만 학습

---

## 실행 순서

### 1. 필요 라이브러리 설치
```bash
conda activate doctor
pip install librosa soundfile scikit-learn matplotlib seaborn tqdm
```

### 2. Coswara 데이터 압축 해제
```bash
cd /home/students/cs/doctor/AI_Doctor/data/Coswara-Data-master

# extract_data.py에 pandas가 필요하므로 설치
pip install pandas

# 압축 해제 실행 (13GB → 약 30GB로 해제)
python extract_data.py
```

**압축 해제 시간**: 약 10~30분 소요

**확인**:
```bash
ls Extracted_data/
# 202* 폴더들이 생성되어 있어야 함
```

### 3. 오디오 전처리
```bash
cd /home/students/cs/doctor/AI_Doctor/test

# 기침/호흡 데이터를 16kHz, 2초 세그먼트로 전처리
python preprocess_coswara_audio.py
```

**출력**:
- `/home/students/cs/doctor/AI_Doctor/data/coswara_processed/`
  - `cough-heavy/normal/` - 정상 기침
  - `cough-heavy/abnormal/` - 비정상 기침 (COVID 양성)
  - `cough-shallow/normal/`
  - `cough-shallow/abnormal/`
  - `breathing-deep/normal/`
  - `breathing-deep/abnormal/`
  - `breathing-shallow/normal/`
  - `breathing-shallow/abnormal/`
  - `metadata.json` - 전체 메타데이터

### 4. HeAR + 분류기 테스트
```bash
# HeAR 임베딩 추출 + 분류기 학습 및 평가
python test_hear_classifier.py
```

**실행 내용**:
1. HeAR 모델 로드 (사전학습된 3억개 데이터)
2. 전처리된 오디오 → HeAR 임베딩 추출 (512차원)
3. Train/Val/Test 분할
4. 간단한 분류기 학습 (2층 fully connected)
5. 테스트 세트 평가
6. 결과 시각화 및 저장

**출력**:
- `results/confusion_matrix.png` - Confusion Matrix
- `results/training_curves.png` - 학습 곡선
- `results/hear_classifier.h5` - 학습된 분류기
- `results/results.json` - 테스트 결과

---

## 예상 결과

### 데이터 통계 (cough-heavy 기준)
| 세트 | 정상 | 비정상 | 총합 |
|------|------|--------|------|
| Train | ~1000 | ~300 | ~1300 |
| Val | ~200 | ~60 | ~260 |
| Test | ~200 | ~60 | ~260 |

### 성능 예측
- **정확도**: 70~85% 예상
- **AUC**: 0.75~0.90 예상

HeAR이 이미 건강 음향 패턴을 학습했기 때문에, 소량의 데이터로도 합리적인 성능을 기대할 수 있습니다.

---

## 파일 설명

### `preprocess_coswara_audio.py`
- Coswara 압축 해제된 데이터 전처리
- 16kHz 리샘플링
- 2초 세그먼트 분할
- 품질 필터링 (excellent만 사용)
- 정상/비정상 라벨링

### `test_hear_classifier.py`
- HeAR 모델 로드 (TFSMLayer)
- 임베딩 추출 (가중치 고정)
- Linear Probing (분류기만 학습)
- 평가 및 시각화

---

## 주의사항

1. **디스크 공간**: 압축 해제 후 약 30GB 필요
2. **GPU 메모리**: HeAR 모델이 GPU를 사용하므로 최소 8GB 권장
3. **실행 시간**:
   - 압축 해제: 10~30분
   - 전처리: 5~10분
   - HeAR 테스트: 10~20분 (GPU 기준)

---

## 문제 해결

### 압축 해제 실패
```bash
# extract_data.py가 동작하지 않으면 수동 해제
cd /home/students/cs/doctor/AI_Doctor/data/Coswara-Data-master
mkdir -p Extracted_data

# 각 날짜별 폴더에서 압축 해제
for dir in 202*; do
    echo "Extracting $dir..."
    cat $dir/*.tar.gz.* | tar -xzf - -C Extracted_data/
done
```

### pandas 모듈 없음
```bash
conda activate doctor
pip install pandas
```

### librosa 모듈 없음
```bash
conda activate doctor
pip install librosa soundfile
```

### GPU 메모리 부족
`test_hear_classifier.py`에서 `batch_size`를 32 → 16 또는 8로 줄이세요.

---

## 다음 단계

1. 호흡 데이터도 테스트 (breathing-deep, breathing-shallow)
2. 기침 + 호흡 데이터 결합하여 앙상블
3. 다른 질환 데이터셋 추가 (결핵, 천식 등)
4. 웹앱/모바일 앱 통합
