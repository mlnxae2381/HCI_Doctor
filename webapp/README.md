# HCI DocTor - AI 의료 진단 보조 웹앱

AI 기반 의료 영상 및 음성 분석 시스템

## 기능

### 1. CT 이미지 분석
- **모델**: MedGemma LoRA (87.35% 정확도)
- **분류**: Normal / Benign / Malignant
- **상태**: 사용 가능 ✅

### 2. X-ray 이미지 분석
- **모델**: MedGemma LoRA
- **분류**: Normal / Abnormal
- **상태**: 학습 중 (42.8% 완료, 3-4일 소요) 🔄

### 3. 기침 소리 분석
- **모델**: HeAR (96.74% 정확도)
- **분류**: Normal / Abnormal
- **상태**: 사용 가능 ✅

### 4. 호흡 소리 분석
- **모델**: HeAR (96.74% 정확도)
- **분류**: Normal / Abnormal
- **상태**: 사용 가능 ✅

### 5. 커뮤니티
- 게시판 (자유게시판, 건강 정보, 질문/답변, 진단 후기)
- 댓글 시스템
- 좋아요 기능

## 사용 기술

### 백엔드
- Flask (Python 웹 프레임워크)
- PyTorch + Transformers (MedGemma)
- TensorFlow (HeAR)
- SQLite (커뮤니티 DB)

### 프론트엔드
- HTML5 + CSS3 + JavaScript
- Font Awesome 아이콘
- 반응형 디자인

### AI 모델
- **MedGemma 1.5-4b-it**: Google의 의료 비전-언어 모델
- **HeAR**: Google의 건강 음향 표현 모델
- **LoRA**: 효율적인 파인튜닝 기법

## 설치 및 실행

### 1. 의존성 설치
```bash
cd /home/students/cs/doctor/AI_Doctor/webapp/backend
pip install -r requirements.txt
```

### 2. 서버 실행
```bash
cd /home/students/cs/doctor/AI_Doctor/webapp
./start.sh
```

또는 직접:
```bash
cd /home/students/cs/doctor/AI_Doctor/webapp/backend
python app.py
```

### 3. 웹 브라우저에서 접속
```
http://localhost:3003
```

## 프로젝트 구조

```
webapp/
├── backend/
│   ├── app.py              # Flask 메인 서버
│   ├── database.py         # 커뮤니티 DB
│   ├── models/
│   │   ├── ct_model.py     # CT 모델
│   │   ├── xray_model.py   # X-ray 모델
│   │   └── audio_model.py  # 오디오 모델
│   └── routes/
│       ├── ct_route.py     # CT API
│       ├── xray_route.py   # X-ray API
│       ├── audio_route.py  # 오디오 API
│       └── community_route.py  # 커뮤니티 API
├── frontend/
│   ├── index.html          # 메인 HTML
│   ├── css/
│   │   └── style.css       # 스타일시트
│   └── js/
│       └── app.js          # JavaScript
├── data/
│   ├── uploads/            # 업로드 파일
│   └── community.db        # 커뮤니티 DB
└── start.sh                # 실행 스크립트
```

## API 엔드포인트

### CT 분석
- `POST /api/ct/analyze` - CT 이미지 분석
- `GET /api/ct/info` - 모델 정보

### X-ray 분석
- `POST /api/xray/analyze` - X-ray 이미지 분석
- `GET /api/xray/info` - 모델 정보

### 오디오 분석
- `POST /api/audio/analyze` - 기침음/호흡음 분석
- `GET /api/audio/info` - 모델 정보

### 커뮤니티
- `GET /api/community/posts` - 게시글 목록
- `POST /api/community/posts` - 게시글 작성
- `GET /api/community/posts/<id>` - 게시글 상세
- `POST /api/community/posts/<id>/like` - 좋아요
- `POST /api/community/posts/<id>/comments` - 댓글 작성

## 주의사항

⚠️ 본 서비스는 의료 전문가의 진단을 대체할 수 없으며, 참고용으로만 사용해야 합니다.

정확한 진단은 반드시 의료 전문가와 상담하시기 바랍니다.

## 라이선스

Educational Use Only
