# AI Doctor 프로젝트 요약

## 프로젝트 개요
- **공모전**: [Kaggle MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge)
- **목표**: MedGemma 및 HAI-DEF 모델을 활용한 의료 AI 앱 개발

---

## 앱 주요 기능 (4가지)

| # | 기능 | 사용 모델 | 상태 |
|---|------|-----------|------|
| 1 | 기침/호흡음 녹음 → 정상/비정상 판단 | **HeAR** | 데이터 다운로드 중 |
| 2 | X-ray 사진 기록 및 분석 | **CXR Foundation** + **MedGemma 1.5** | 다른 서버에서 가공 완료 |
| 3 | CT 사진으로 폐암 진단 | **MedGemma 1.5** | 데이터 가공 진행 중 |
| 4 | 커뮤니티 (회원가입/로그인) | 백엔드 개발 필요 | 미착수 |

---

## 사용 모델

### MedGemma 1.5 (메인 모델)
- 모델 ID: `google/medgemma-1.5-4b-it`
- 멀티모달 지원: 2D 이미지 (X-ray, 피부 등) + **3D 이미지 (CT, MRI)**
- 텍스트 Q&A 지원

### HAI-DEF 모델 컬렉션
| 모델 | 용도 | Hugging Face |
|------|------|--------------|
| HeAR | 건강 음향 분석 (기침, 호흡음) | google/hear |
| CXR Foundation | 흉부 X-ray 임베딩 | google/cxr-foundation |
| Path Foundation | 병리 슬라이드 임베딩 | google/path-foundation |
| Derm Foundation | 피부 이미지 임베딩 | google/derm-foundation |
| MedSigLIP | 의료 이미지-텍스트 임베딩 | MedGemma 내장 |

> **참고**: CT-Foundation은 별도 모델이 아니며, MedGemma 1.5에 통합됨

---

## CT 데이터셋: IQ-OTH/NCCD Lung Cancer Dataset

### 출처
- 이라크 종양학 교육병원 / 국립 암질환센터
- 2019년 가을 수집
- 종양학자 + 방사선 전문의 라벨링

### 데이터 구조
```
AI_Doctor/data/Lung Cancer Dataset/
├── Test cases/                           → 197장
└── The IQ-OTHNCCD lung cancer dataset/
    └── The IQ-OTHNCCD lung cancer dataset/
        ├── Bengin cases/                 → 120장 (15명)
        ├── Malignant cases/              → 561장 (40명)
        └── Normal cases/                 → 416장 (55명)
```

### 데이터 통계
| 클래스 | 이미지 수 | 환자 수 |
|--------|-----------|---------|
| Normal (정상) | 416장 | 55명 |
| Benign (양성) | 120장 | 15명 |
| Malignant (악성) | 561장 | 40명 |
| Test cases | 197장 | - |
| **총합** | **1,294장** | 110명 |

### 데이터 특징
- 형식: JPG (원본은 DICOM)
- 스캐너: SOMATOM (Siemens)
- 슬라이스 두께: 1mm
- 환자당 슬라이스: 80~200장

### 데이터셋 평가
**장점**
- 종양학자 + 방사선 전문의 라벨링
- 3-class 구성 (Normal/Benign/Malignant) 명확
- 실제 병원 임상 데이터

**단점**
- 데이터 양 부족 (1,294장)
- 클래스 불균형: Benign(120) vs Malignant(561)
- 2D 슬라이스 (MedGemma 1.5는 3D 지원)

---

## 데이터 전처리 계획

### 1단계: 디렉토리 구조
```
AI_Doctor/data/processed/
├── train/
│   ├── normal/
│   ├── benign/
│   └── malignant/
├── val/
├── test/
├── train.json
├── val.json
└── test.json
```

### 2단계: Train/Val/Test 분할
| 세트 | 비율 |
|------|------|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

### 3단계: 데이터 증강 (클래스 불균형 해결)
| 클래스 | 현재 | 목표 |
|--------|------|------|
| Normal | 416장 | ~600장 |
| Benign | 120장 | **~600장** (5배 증강) |
| Malignant | 561장 | ~600장 |

### 증강 기법
- 회전: 90°, 180°, 270°
- 플립: 좌우, 상하
- 밝기 조절: ±20%
- 대비 조절

### 4단계: MedGemma 파인튜닝 포맷 (JSON)
```json
{
  "id": "image_id",
  "image": "train/normal/normal_0001.png",
  "label": "normal",
  "conversations": [
    {
      "role": "user",
      "content": "이 CT 이미지를 분석하여 폐암 여부를 판단해주세요."
    },
    {
      "role": "assistant",
      "content": "이 CT 이미지는 정상 소견입니다..."
    }
  ]
}
```

---

## 파일 구조

```
AI_Doctor/
├── data/
│   ├── Lung Cancer Dataset/          # 원본 CT 데이터
│   ├── processed/                    # 전처리된 데이터 (생성 예정)
│   └── preprocess_ct_data.py         # 전처리 스크립트
├── test/
│   ├── test_haidef_all.py            # HAI-DEF 모델 테스트
│   └── test_medgemma.py              # MedGemma 테스트
├── requirements.txt
└── PROJECT_SUMMARY.md                # 이 파일
```

---

## 다음 단계 (TODO)

- [ ] CT 데이터 전처리 스크립트 실행 완료
- [ ] 기침음 데이터 다운로드 완료 후 가공
- [ ] MedGemma 1.5 파인튜닝
- [ ] HeAR 모델 파인튜닝
- [ ] 앱 백엔드/프론트엔드 개발
- [ ] 커뮤니티 기능 구현

---

## 참고 링크

- [HAI-DEF Collection](https://huggingface.co/collections/google/health-ai-developer-foundations-hai-def)
- [MedGemma 1.5](https://huggingface.co/google/medgemma-1.5-4b-it)
- [MedGemma Blog](https://research.google/blog/next-generation-medical-image-interpretation-with-medgemma-15-and-medical-speech-to-text-with-medasr/)
