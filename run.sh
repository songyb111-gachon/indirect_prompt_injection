#!/bin/bash
echo "======================================================"
echo "Indirect Prompt Injection (From Image) 실습"
echo "======================================================"

echo "[1/2] 필요한 패키지 확인 중..."
pip install -r requirements.txt -q

echo "[2/2] 실험 실행 중..."
python image_perturbation_attack.py --demo --output results

echo "======================================================"
echo "실험 완료!"
echo "======================================================"

