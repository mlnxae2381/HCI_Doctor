"""
X-ray 테스트 진행 상황 모니터링
"""
import time
import os
import sys

log_file = "/home/students/cs/doctor/AI_Doctor/test/xray_test_output.log"
total_samples = 11213

print("=" * 60)
print("X-ray LoRA 테스트 진행 상황 모니터링")
print("=" * 60)
print(f"전체 샘플 수: {total_samples:,}개")
print(f"로그 파일: {log_file}")
print()

try:
    while True:
        # 로그 파일에서 "진행:" 패턴 찾기
        with open(log_file, 'r') as f:
            content = f.read()

        # 파일 크기
        file_size = os.path.getsize(log_file) / 1024  # KB

        # "진행: X/Y" 패턴 찾기
        import re
        progress_matches = re.findall(r'진행: (\d+)/(\d+)', content)

        if progress_matches:
            current, total = map(int, progress_matches[-1])
            percentage = (current / total) * 100
            print(f"\r진행: {current}/{total} ({percentage:.1f}%) | 로그: {file_size:.1f} KB", end='', flush=True)
        else:
            print(f"\r로그 크기: {file_size:.1f} KB (초기화 중...)", end='', flush=True)

        # 완료 체크
        if "테스트 평가 완료" in content or "✨" in content:
            print("\n\n테스트 완료!")
            break

        time.sleep(5)

except KeyboardInterrupt:
    print("\n\n모니터링 종료")
    sys.exit(0)
