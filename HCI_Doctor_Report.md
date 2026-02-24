# HCI DocTor: AI 기반 의료 진단 보조 웹 애플리케이션
## Kaggle MedGemma Impact Challenge 출품작

---

## 1. 프로젝트 기획 배경

### 1.1 한국의 폐암 환자 온라인 커뮤니티 현상

한국 최대 포털 사이트인 네이버(Naver)에는 '카페(Cafe)'라는 커뮤니티 서비스가 있다. 이 중에는 폐암 환자들이 모인 카페가 존재하며, 이 카페에서는 매우 특이한 현상이 관찰된다. 환자들이 자신의 CT 사진이나 흉부 X-ray 사진을 직접 업로드하고, "이 사진에서 폐암 증상이 보이나요?"라는 질문을 남기는 것이다. 이에 대해 다른 환자들, 때로는 실제 의사들까지 댓글로 의견을 나누고 있다.

이러한 현상이 발생하는 근본적인 원인은 **진단 대기 시간**에 있다. 한국의 의료 시스템에서 CT 검사를 받고 공식적인 판독 결과를 받기까지 수 주에서 수 개월이 걸리는 경우가 많다. 암이 의심되는 상황에서 그 긴 시간 동안 아무런 정보 없이 기다려야 하는 환자들의 불안감은 매우 크다. 그 결과, 환자들은 전문적인 의료 판단을 대신할 수 있는 어떤 정보라도 얻으려 온라인 커뮤니티에 의존하게 된 것이다.

### 1.2 문제 인식

우리는 이 현상에서 두 가지 문제를 발견했다.

첫째, **의료 정보의 불확실성**: 비전문가나 타 환자의 의견은 정확한 의학적 판단을 대체할 수 없으며, 잘못된 정보가 환자의 불안을 오히려 가중시킬 수 있다.

둘째, **충족되지 않은 수요의 존재**: 역설적으로, 이 현상은 환자들이 얼마나 빠른 진단 보조 도구를 원하는지를 명확히 보여준다. 수십만 명이 이러한 커뮤니티를 이용한다는 사실은 AI 기반 진단 보조 도구에 대한 강력한 현실적 수요가 존재함을 의미한다.

### 1.3 솔루션 기획

우리는 이 네이버 카페를 벤치마킹하여, 환자들이 실제로 원하는 기능을 AI로 제공하는 웹 애플리케이션을 기획했다. 핵심 아이디어는 다음과 같다.

- CT/X-ray 이미지를 업로드하면 AI가 즉시 분석 결과를 제공
- 기침이나 호흡음을 분석하여 폐 기능 이상 여부를 판단
- 커뮤니티 기능을 통해 환자들이 정보를 공유하는 공간 제공

기술 기반으로는 Google의 의료 AI 모델 컬렉션인 **HAI-DEF(Health AI Developer Foundations)**를 활용했다. 특히 의료 이미지 분석에 특화된 **MedGemma 1.5**, 호흡음 분석에 특화된 **HeAR(Health Acoustic Representations)** 모델을 중심으로 개발했다.

---

## 2. 사용 기술 스택 및 모델

### 2.1 AI 모델

| 모델 | 용도 | 출처 |
|------|------|------|
| MedGemma 1.5 4B IT | CT 및 X-ray 이미지 분석 | Google (HuggingFace: `google/medgemma-1.5-4b-it`) |
| HeAR | 기침/호흡음 분석 | Google (HuggingFace: `google/hear`) |
| PEFT LoRA | MedGemma 파인튜닝 기법 | HuggingFace `peft` 라이브러리 |

**MedGemma 1.5**는 Google이 의료 이미지와 텍스트를 동시에 처리할 수 있도록 설계한 멀티모달 LLM이다. 흉부 X-ray, CT 등 의료 이미지를 입력으로 받아 자연어로 분석 결과를 생성한다.

**HeAR**은 기침, 호흡음 등 건강 관련 음향 데이터에 특화된 임베딩 모델로, 오디오를 고차원 특징 벡터로 변환한다.

### 2.2 데이터셋

| 데이터셋 | 크기 | 용도 |
|----------|------|------|
| IQ-OTH/NCCD Lung Cancer Dataset | 1,294장 (CT JPG) | CT 모델 파인튜닝 |
| NIH Chest X-ray 14 | ~112,000장 | X-ray 모델 파인튜닝 |
| Coswara Dataset | ~2,000개 오디오 | HeAR 분류기 학습 |
| ICBHI 2017 | 기관지음 wav 파일 | HeAR 분류기 학습 |

### 2.3 개발 스택

- **백엔드**: Python Flask (REST API)
- **프론트엔드**: Vanilla JavaScript, History API 기반 SPA(Single Page Application)
- **GPU 환경**: NVIDIA GPU 서버 (GPU 4개 — CT용 GPU 0, X-ray용 GPU 1, HeAR용 GPU 2)
- **딥러닝 프레임워크**: PyTorch (MedGemma), TensorFlow (HeAR)

---

## 3. 데이터 전처리

### 3.1 CT 데이터 전처리

IQ-OTH/NCCD 데이터셋은 이라크 종양학 교육병원에서 수집된 실제 임상 데이터로, SOMATOM(Siemens) CT 스캐너로 촬영된 이미지를 JPG로 변환한 것이다.

**원본 데이터 구성**
- Normal(정상): 416장 (55명)
- Benign(양성 종양): 120장 (15명)
- Malignant(악성 종양): 561장 (40명)

**전처리 과정**
1. Train / Validation / Test 세트 분할 (70% / 15% / 15%)
2. **클래스 불균형 처리**: Benign 클래스가 120장으로 다른 클래스 대비 극도로 적었다. 이를 해결하기 위해 회전(90°/180°/270°), 좌우 플립, 밝기 조절 등의 데이터 증강(Data Augmentation)을 적용하여 Benign 샘플을 약 5배 증강했다.
3. MedGemma의 입력 포맷에 맞게 JSON 형태의 대화 데이터로 변환:
   - 입력: CT 이미지 + "이 CT 이미지를 분석하여 병변이 정상인지, 양성 종양인지, 악성 종양인지 판단해주세요."
   - 출력: "이 CT 이미지는 정상입니다." / "양성 종양 소견이 관찰됩니다." / "악성 종양이 의심됩니다."

### 3.2 X-ray 데이터 전처리

NIH Chest X-ray 14 데이터셋을 Normal(정상)과 Abnormal(비정상) 2개 클래스로 재분류했다.

- Normal: 6,021개 (테스트 세트 기준)
- Abnormal: 5,192개 (테스트 세트 기준)

X-ray 데이터 역시 CT와 동일한 대화 포맷으로 변환하여 MedGemma 파인튜닝에 사용했다.

### 3.3 오디오 데이터 전처리

Coswara 데이터셋은 호흡 방식(깊게/얕게)과 기침 방식(강하게/약하게)으로 구분된 오디오 파일들을 포함한다. 전처리 과정:

1. HeAR 모델로 각 오디오 파일을 1280차원 임베딩 벡터로 변환
2. 변환된 벡터를 numpy 배열로 저장
3. 저장된 임베딩 위에 3-레이어 MLP(Multi-Layer Perceptron) 분류기를 학습

**심각한 클래스 불균형 문제**: Abnormal 885개, Normal 35개 (약 25:1 비율). 이를 해결하기 위해 클래스 가중치 조정(class_weight='balanced')을 적용했다.

---

## 4. 모델 파인튜닝

### 4.1 MedGemma LoRA 파인튜닝 방법론

MedGemma 1.5 4B 모델 전체를 파인튜닝하는 것은 메모리와 연산 자원 면에서 비현실적이다. 따라서 **LoRA(Low-Rank Adaptation)** 기법을 사용했다.

LoRA는 원본 모델의 가중치를 고정(freeze)한 상태에서 각 레이어에 소규모의 학습 가능한 행렬(rank decomposition matrix)을 추가하는 방식이다. 이를 통해 전체 모델 파라미터의 극히 일부만 학습하면서도 특정 도메인에 대한 성능을 효과적으로 향상시킬 수 있다.

**파인튜닝 설정**
- 기반 모델: `google/medgemma-1.5-4b-it`
- LoRA rank: 16
- LoRA alpha: 32
- 학습률: 2e-4
- 배치 크기: bfloat16 혼합 정밀도 학습
- 옵티마이저: AdamW
- 그래디언트 체크포인팅 활성화 (메모리 절약)

### 4.2 CT 모델 학습 결과

- 최종 Val Loss: 0.0787
- **테스트 정확도: 87.35%** (166개 샘플)
- Malignant 정밀도: 95.5%, Recall: 98.8%
- Benign 성능은 상대적으로 낮음 (클래스 불균형의 영향)

### 4.3 X-ray 모델 학습 결과

- 체크포인트: step-3600, Val Loss: 0.0929
- **테스트 정확도: 72.38%** (11,213개 샘플)
- Normal: precision 0.74, recall 0.74, F1 0.74
- Abnormal: precision 0.70, recall 0.70, F1 0.70

### 4.4 HeAR 오디오 분류기 학습 결과

- **테스트 정확도: 96.74%** (184개 샘플)
- HeAR 임베딩 위에 학습된 MLP 분류기

---

## 5. 개발 과정에서 겪은 주요 문제와 해결

### 5.1 문제: GPU OOM(Out of Memory) 오류

**증상**: MedGemma 4B 모델을 로드할 때 CUDA Out of Memory 오류 발생.

**원인**: 4B 파라미터 모델을 float32로 로드하면 약 16GB 이상의 GPU 메모리가 필요하다.

**해결**: `bfloat16` 혼합 정밀도(Mixed Precision)를 사용하여 메모리 사용량을 절반으로 줄였다. 또한 그래디언트 체크포인팅(Gradient Checkpointing)을 활성화하여 역전파 시 메모리 효율을 높였다.

```python
base_model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
```

### 5.2 문제: 프롬프트 오염(Prompt Contamination)에 의한 잘못된 분류

**증상**: 모델이 "이 CT 이미지는 정상입니다. 폐암을 시사하는 소견이 관찰되지 않습니다."라고 응답했음에도 불구하고 시스템이 'Malignant(악성)'으로 분류했다.

**원인 분석**: MedGemma는 텍스트 생성 시 입력 프롬프트와 출력 텍스트를 이어붙인 전체 시퀀스를 디코딩한다. 즉, 모델의 출력 텍스트에는 입력 프롬프트인 "이 CT 이미지를 분석하여 **병변이 정상인지, 양성 종양인지, 악성 종양인지** 판단해주세요."가 그대로 포함되어 있다.

라벨 추출 로직은 텍스트 내 키워드를 점수화하는 방식이었는데, 프롬프트 자체에 포함된 "양성 종양"(+6점), "악성 종양"(+7점), "폐암"(+6점) 등의 키워드가 점수를 오염시켜, 실제 응답이 "정상"이라 해도 Malignant로 잘못 분류되었다.

```
프롬프트에 포함된 "양성 종양"  → benign  +6
프롬프트에 포함된 "악성 종양"  → malignant +7
응답에 포함된 "폐암...관찰되지 않" → normal +4
결과: malignant(13) > normal(4)  → 잘못된 판단
```

**해결**: 라벨 추출 시 프롬프트 종료 키워드("판단해주세요.", "분석해주세요." 등)를 기준으로 텍스트를 분할하여, 프롬프트 이후의 **응답 부분만** 키워드 분석에 사용하도록 수정했다.

```python
def extract_label(self, text):
    rt = text.lower().strip()
    for ending in ['판단해주세요.', '평가해주세요.', '분석해주세요.']:
        idx = rt.find(ending)
        if idx != -1:
            rt = rt[idx + len(ending):].strip()  # 응답 부분만 추출
            break
    # 이후 rt만 사용하여 점수 계산
```

### 5.3 문제: 신뢰도(Confidence) 수치 비정상 표시 (500%, 1300%)

**증상**: 프론트엔드에서 신뢰도가 "Benign 600.0% / Malignant 1300.0% / Normal 500.0%"와 같이 비정상적으로 표시되었다.

**원인**: 라벨 추출 로직은 각 클래스에 키워드 매칭 점수(정수값: 5, 6, 13 등)를 부여하는 방식이었다. 그런데 백엔드 API가 이 정수 점수를 그대로 `confidence` 필드에 담아 반환했고, 프론트엔드는 이 값이 0~1 사이의 확률값이라고 가정하고 100을 곱해 퍼센트로 변환했다.

**해결**: 백엔드 `predict()` 함수에서 점수를 반환하기 전에, 전체 점수 합으로 나누어 0~1 범위의 확률값으로 정규화하도록 수정했다.

```python
total = sum(scores.values())
if total > 0:
    confidence = {k: round(v / total, 4) for k, v in scores.items()}
else:
    confidence = {k: 1/3 for k in scores}  # 기본값
```

### 5.4 문제: 모델 어댑터 경로 오류

**증상**: 서버 시작 시 `"Repo id must be in the form 'namespace/repo_name': '/home/.../ct_lora_adapter'"` 오류 발생.

**원인**: `ct_model.py`에 설정된 `ADAPTER_PATH`가 실제 어댑터가 저장된 경로와 달랐다. HuggingFace의 `from_pretrained()` 함수는 로컬 경로가 존재하지 않으면 해당 문자열을 HuggingFace Hub의 원격 저장소 ID로 해석하려 시도하는데, 존재하지 않는 경로이므로 위 오류가 발생했다.

**해결**: `ADAPTER_PATH`를 실제 어댑터가 저장된 경로로 수정했다.

### 5.5 문제: X-ray 라벨 추출 정확도 한계

**증상**: 초기 X-ray 모델 정확도가 67.21%로, CT 모델(87.35%)에 비해 낮았다.

**원인 분석**:
1. 프롬프트 오염 문제 (5.2와 동일)
2. 키워드 부재 처리: "관찰되지 않습니다"와 같은 부정 표현이 일부 정상 키워드로 인식되지 않는 경우
3. "정상 폐"와 같은 정상 신호 키워드 누락

**해결**: 라벨 추출 로직을 v1에서 v2로 개선했다. `reeval_xray_labels.py` 스크립트를 통해 11,213개 샘플의 생성 텍스트를 모델 재추론 없이 재평가함으로써 빠르게 로직을 비교·검증했다. 개선 후 최종 정확도 **72.38%**를 달성했다.

---

## 6. 최종 시스템 구조

### 6.1 전체 아키텍처

```
사용자 브라우저
    │
    ▼
[Frontend - Vanilla JS SPA]
    - History API 라우터
    - 이미지/오디오 드래그&드롭 업로드
    - 분석 결과 시각화
    │  POST /api/ct/analyze
    │  POST /api/xray/analyze
    │  POST /api/audio/analyze
    ▼
[Backend - Flask API Server, port 3003]
    │
    ├── CT Route  → CTModel (MedGemma + LoRA) → GPU 0
    ├── X-ray Route → XrayModel (MedGemma + LoRA) → GPU 1
    ├── Audio Route → AudioModel (HeAR + MLP) → GPU 2
    └── Community Route → SQLite DB
```

### 6.2 분석 흐름

1. 사용자가 이미지(CT/X-ray) 또는 오디오 파일 업로드
2. Flask 서버가 파일을 임시 저장 후 해당 모델에 전달
3. MedGemma: 이미지 + 프롬프트를 입력받아 자연어 응답 생성
4. 응답 텍스트에서 키워드 기반 점수 계산 → 확률로 정규화
5. 분류 결과(Normal/Benign/Malignant 또는 Normal/Abnormal)와 신뢰도, 모델 설명 반환
6. 프론트엔드에서 결과 시각화

### 6.3 주요 기능

| 기능 | 설명 |
|------|------|
| CT 폐암 진단 | CT 이미지 업로드 → Normal / Benign / Malignant 분류 |
| 흉부 X-ray 분석 | X-ray 이미지 업로드 → Normal / Abnormal 분류 |
| 호흡음/기침 분석 | 오디오 파일 업로드 → Normal / Abnormal 분류 |
| 커뮤니티 | 회원가입, 로그인, 게시글/댓글 CRUD |

---

## 7. 성과 및 결론

### 7.1 최종 모델 성능

| 모델 | 정확도 | 테스트 샘플 수 | 분류 |
|------|--------|--------------|------|
| CT 폐암 진단 (MedGemma LoRA) | **87.35%** | 166개 | Normal / Benign / Malignant |
| 흉부 X-ray 분석 (MedGemma LoRA) | **72.38%** | 11,213개 | Normal / Abnormal |
| 호흡음 분석 (HeAR + MLP) | **96.74%** | 184개 | Normal / Abnormal |

### 7.2 의의

본 프로젝트는 단순히 AI 모델을 구현하는 것을 넘어, 실제 사용자 니즈에서 출발한 솔루션을 제시했다는 점에서 의의가 있다. 네이버 카페 현상이 보여주듯, 환자들은 이미 AI 기반 즉각적 진단 보조에 대한 강한 수요를 가지고 있다. HCI DocTor는 Google의 HAI-DEF 모델 컬렉션을 활용하여 이 수요에 대응하는 실용적인 서비스를 구현했다.

또한 MedGemma의 멀티모달 능력과 HeAR의 음향 분석 능력을 단일 플랫폼에 통합함으로써, CT/X-ray 이미지 분석과 호흡음 분석을 하나의 앱에서 제공하는 종합적인 폐 건강 보조 서비스를 실현했다.

### 7.3 한계 및 향후 과제

- X-ray 정확도 72.38%는 임상 활용을 위해 추가 개선이 필요하다. 더 많은 학습 데이터 또는 더 정교한 프롬프트 설계로 향상 가능하다고 판단한다.
- 본 서비스는 의료 진단을 대체하는 것이 아닌, 진단 대기 기간 동안 환자의 불안을 줄여주는 **보조 도구**임을 명확히 한다.
- 향후 DICOM 포맷 직접 지원, 슬라이스 단위 3D CT 분석 등으로 확장 가능하다.

---

## 참고

- MedGemma: https://huggingface.co/google/medgemma-1.5-4b-it
- HeAR: https://huggingface.co/google/hear
- HAI-DEF Collection: https://huggingface.co/collections/google/health-ai-developer-foundations-hai-def
- IQ-OTH/NCCD Dataset: https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset
- Coswara Dataset: https://github.com/iiscleap/Coswara-Data
- GitHub: https://github.com/mlnxae2381/HCI_Doctor
