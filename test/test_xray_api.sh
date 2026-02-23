#!/bin/bash
# X-ray API 테스트 스크립트

echo "================================================"
echo "X-ray API 테스트"
echo "================================================"

# 1. 모델 정보 확인
echo -e "\n[1] X-ray 모델 정보 확인"
curl -s http://localhost:3003/api/xray/info | python3 -m json.tool

# 2. Health check
echo -e "\n\n[2] Health Check"
curl -s http://localhost:3003/health | python3 -m json.tool

echo -e "\n\n================================================"
echo "테스트 이미지로 분석 테스트를 하려면:"
echo "  curl -X POST -F \"file=@/path/to/xray.jpg\" http://localhost:3003/api/xray/analyze"
echo "================================================"
