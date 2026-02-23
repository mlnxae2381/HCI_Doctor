# HCI DocTor - 라우팅 시스템 가이드

## 📋 구현 완료 내역

### ✅ 백업
- **전체 백업**: `/home/students/cs/doctor/AI_Doctor/webapp/frontend_backup_20260215_002953/`
- **app.js 백업**: `frontend/js/app.js.backup_*`

### ✅ 새로 생성된 파일
1. **`frontend/js/router.js`** - History API 기반 라우터 시스템
2. **`frontend/js/pages.js`** - 페이지 컴포넌트 및 렌더링 함수

### ✅ 수정된 파일
1. **`frontend/js/app.js`** - 라우터 통합 및 초기화
2. **`frontend/index.html`** - 스크립트 로드 순서 및 네비게이션 링크 업데이트
3. **`backend/app.py`** - SPA 라우팅 지원 (404 핸들러, 정적 파일 서빙)

---

## 🛣️ 라우트 맵핑

| 번호 | 경로 | 페이지 | 상태 |
|------|------|--------|------|
| 01 | `/` | 메인 대시보드 (홈) | ✅ 완료 (기존) |
| 02 | `/community` | 커뮤니티 카테고리 | ✅ 완료 (기존) |
| 03 | `/community/feed` | 커뮤니티 상세 피드 | ⏳ 임시 컴포넌트 |
| 04 | `/login` | 로그인 | ✅ 완료 (모달) |
| 05 | `/signup` | 회원가입 | ✅ 완료 (모달) |
| 06 | `/about` | 소개 | ✅ 완료 (기존) |
| 07 | `/analysis/ct` | CT 상세 | ⏳ 임시 컴포넌트 |
| 08 | `/analysis/xray` | X-ray 상세 | ⏳ 임시 컴포넌트 |
| 09 | `/analysis/acoustic` | 호흡음 상세 | ⏳ 임시 컴포넌트 |
| - | `/404` | 404 페이지 | ✅ 완료 |

---

## 🔧 기존 기능 유지 확인

### ✅ 100% 유지된 기능
- **CT 이미지 분석** (API 연동)
- **X-ray 이미지 분석** (API 연동)
- **기침음 분석** (API 연동)
- **호흡음 분석** (API 연동)
- **커뮤니티 게시판** (CRUD)
- **로그인/회원가입** (UI)
- **모달 시스템** (모든 분석 모달)

---

## 🚀 실행 방법

### 1. 서버 시작
```bash
cd /home/students/cs/doctor/AI_Doctor/webapp/backend
python app.py
```

### 2. 브라우저에서 접속
```
http://localhost:3003
```

### 3. 라우팅 테스트
브라우저 주소창에서 직접 입력하여 테스트:
- `http://localhost:3003/` - 홈
- `http://localhost:3003/community` - 커뮤니티
- `http://localhost:3003/about` - 소개
- `http://localhost:3003/login` - 로그인
- `http://localhost:3003/analysis/ct` - CT 분석 (임시)
- `http://localhost:3003/analysis/acoustic` - 음향 분석 (임시)

---

## 📝 임시 페이지 설명

다음 페이지들은 **임시 컴포넌트**로 구성되어 있습니다:
- `/community/feed`
- `/analysis/ct`
- `/analysis/xray`
- `/analysis/acoustic`

### 임시 페이지 특징
- ✅ **기존 기능 로직 100% 유지** (버튼 클릭 시 기존 모달 실행)
- ✅ **라우팅 동작** (URL 변경 및 히스토리 관리)
- ✅ **명확한 상태 표시** (UI 디자인 대기 중 안내)
- ⏳ **UI 디자인 미완성** (추후 업데이트 예정)

---

## 🔍 작동 원리

### History API 기반 라우팅
1. **링크 클릭** → `data-link` 속성 감지
2. **URL 변경** → `history.pushState()` 사용
3. **라우트 매칭** → 정확한 경로 또는 패턴 매칭
4. **페이지 렌더링** → 해당 페이지 함수 실행
5. **뒤로/앞으로** → `popstate` 이벤트 처리

### Flask 백엔드 지원
- **API 요청** (`/api/*`) → JSON 응답
- **정적 파일** (`/css/*`, `/js/*`) → 파일 서빙
- **기타 경로** → `index.html` 반환 (SPA 라우팅)

---

## 🎯 다음 단계

### 1. UI 디자인 완성
임시 페이지들의 UI 디자인 완료 후:
```javascript
// pages.js에서 해당 함수 수정
function renderCTAnalysisPage() {
    // 실제 UI 구현
}
```

### 2. 추가 라우트 등록
```javascript
// app.js에 추가
router.addRoute('/새경로', render새페이지);
```

### 3. 동적 라우트 사용
```javascript
// 패턴 매칭 지원
router.addRoute('/analysis/:type', (params) => {
    console.log(params.type); // 'ct', 'xray' 등
});
```

---

## ⚠️ 주의사항

1. **브라우저 새로고침**: 모든 경로에서 정상 작동 (Flask가 index.html 반환)
2. **API 호출**: 기존 방식 그대로 유지 (`/api/*` 경로)
3. **모달 기능**: 기존 기능 100% 유지
4. **데이터 상태**: 페이지 전환 시 초기화됨 (필요시 상태 관리 추가)

---

## 🐛 트러블슈팅

### 문제: 페이지가 로드되지 않음
**해결**: 브라우저 콘솔에서 스크립트 로드 확인
```javascript
console.log(router); // router 객체 확인
```

### 문제: 정적 파일(CSS/JS) 404
**해결**: Flask 서버 재시작
```bash
cd backend && python app.py
```

### 문제: API 호출 실패
**해결**: API 경로 확인 (`/api/` 로 시작하는지)

---

## 📞 문의

작업 완료: 2026-02-15
백업 위치: `/home/students/cs/doctor/AI_Doctor/webapp/frontend_backup_20260215_002953/`
