#!/bin/bash

echo "=============================================="
echo "HCI DocTor - AI 의료 진단 보조 웹앱 시작"
echo "=============================================="
echo ""

# 가상환경 활성화 (doctor 환경 사용)
source ~/.conda/envs/doctor/bin/activate

# 백엔드 디렉토리로 이동
cd /home/students/cs/doctor/AI_Doctor/webapp/backend

# 서버 실행
echo "서버 시작: http://localhost:3003"
echo "Ctrl+C로 종료할 수 있습니다."
echo ""

python app.py
