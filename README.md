# Indirect Prompt Injection (From Image) 실습

Vision-Language 모델에 대한 이미지 기반 간접 프롬프트 인젝션 공격 실습

## 개요

이미지에 보이지 않는 perturbation을 추가하여, Vision-Language 모델이 숨겨진 프롬프트를 인식하고 따르도록 만드는 공격입니다.

## 실행 방법

```bash
# Linux 서버
chmod +x run.sh
./run.sh

# 또는 직접 실행
pip install -r requirements.txt
python image_perturbation_attack.py --demo
```

## 파일 구조

```
indirect_prompt_injection/
├── image_perturbation_attack.py  # 메인 공격 코드
├── requirements.txt              # 의존성
├── run.sh                       # 실행 스크립트
└── README.md                    # 설명서
```

## 참고

- https://github.com/ebagdasa/multimodal_injection
- LLaVA: https://github.com/haotian-liu/LLaVA

