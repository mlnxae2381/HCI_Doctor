import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
# 임베딩 기반 질환 분류 및 정확도 측정
# 이 파일은 step2에서 추출한 임베딩을 사용하여 간단한 분류 모델을 학습하고 테스트하는 과정
def run_performance_test():
    # 1. step2에서 저장한 데이터 불러오기
    LOAD_PATH = "/home/students/cs/doctor/AI_Doctor/results/breath_embeddings.npy"
    
    if not os.path.exists(LOAD_PATH):
        print(f"❌ 임베딩 파일을 찾을 수 없습니다: {LOAD_PATH}\nstep2를 먼저 완료해주세요.")
        return

    print("📂 추출된 임베딩 로드 중...")
    data = np.load(LOAD_PATH, allow_pickle=True).item()
    X = data['embeddings']      # (N, 512) 형태의 숫자들
    filenames = data['filenames'] # 파일명 리스트

    # 2. 정답 데이터(Label) 매칭 (가장 중요한 부분!)
    # 파일명에 'normal'이나 'abnormal'이 포함되어 있다고 가정하고 라벨을 만듭니다.
    # 본인의 파일명 규칙에 맞게 이 부분을 수정해야 합니다.
   
    y = []
    for f in filenames:
        # 경로 문자열 전체를 소문자로 바꿔서 검사
        full_path = f.lower()
        
        if 'abnormal' in full_path:
            y.append(1)  # 비정상
        elif 'normal' in full_path:
            y.append(0)  # 정상
        else:
            # 둘 다 해당 안 될 경우를 대비해 파일 경로 출력해보기
            print(f"❓ 라벨을 결정할 수 없는 파일: {f}")
            y.append(0) # 기본값
    
    y = np.array(y)
    print(f"✅ 데이터 로드 완료: {len(X)} 샘플 (정상: {sum(y==0)}, 비정상: {sum(y==1)})")

    # 3. 학습 데이터와 테스트 데이터 분리 (8:2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. 경량 분류기 학습 (HeAR 모델의 성능을 테스트하는 가장 표준적인 방법)
    print("🧠 분류 모델 학습 및 테스트 중...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # 5. 결과(정확도) 출력
    y_pred = clf.predict(X_test)
    
    print("\n" + "=" * 40)
    print(f"📊 최종 모델 테스트 결과 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
    print("=" * 40)
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal']))

if __name__ == "__main__":
    run_performance_test()