# 실행 / 설치 매뉴얼

### 프로젝트명: 데이터 이질성에 강인한 연합학습을 위한 FedFM 기법 분석 및 구현
### 구성: Flower 기반 연합학습 서버 + 클라이언트
### 언어 / 프레임워크: Python, PyTorch, Flower


## 1. 실행환경

### 1.1 운영체제
- Windows / Linux / macOS 등
- 클라이언트는 각기 다른 환경에서도 실행 가능
### 1.2 필수 소프트웨어
- Python 3.8 이상
- pip
- 인터넷 연결 (서버-클라이언트 통신)
### 1.3 사용 라이브러리
- torch
- torchvision
- flwr
- numpy
- opencv-python
- scikit-learn


## 2. 프로젝트 구조
```
src/
├─ kvasir-seg/
│  └─ Kvasir-SEG/
│     ├─ images/                 # Kvasir-SEG 원본 이미지 데이터
│     ├─ masks/                  # 이미지와 1:1 대응되는 segmentation 마스크
│     └─ kvasir_bboxes.json      # 객체 위치 정보(Bounding Box 메타데이터)
│
├─ Client_code_1.ipynb           # 연합학습 클라이언트 1 실행 노트북
├─ Client_code_2.ipynb           # 연합학습 클라이언트 2 실행 노트북
├─ Client_code_3.ipynb           # 연합학습 클라이언트 3 실행 노트북
│
├─ server.py                     # Flower 기반 연합학습 서버 코드
│
├─ 서버 로그1.png                # 서버 실행 로그 캡처 이미지
└─ 서버 로그2.png                # 서버 학습 진행 로그 캡처 이미지
```


## 3. 데이터셋 준비

### 3.1 데이터셋
- Kvasir-SEG 위장관 용종 분할 데이터셋(약 1000장)
### 3.2 폴더 규칙
- 이미지와 마스크는 파일명이 정확히 일치해야 함


## 4. 라이브러리 설치
`pip install torch torchvision flwr numpy opencv-python scikit-learn`


## 5. 서버 실행 방법

### 5.1 서버 주소 설정
`server.py` 내부에서 서버 주소 및 포트 확인
- 기본 포트: `8080`
- 대기 주소: `0.0.0.0:8000`
### 5.2 서버 실행
`python server.py`
### 5.3 서버 동작 설명
- 최소 클라이언트 수(`min_available_clients=3`) 충족 시 학습 시작
- 총 라운드 수: `num_rounds=5`
- 각 라운드마다 클라이언트에 학습 설정 전달
- 종료 시 라운드별 loss summary 출력 후 자동 종료


## 6. 클라이언트 실행 방법

### 6.1 클라이언트 설정
`client.py`에서 다음 항목 확인
- `CLIENT_ID`
- `SERVER_ADDRESS`
- 데이터 경로
### 6.2 클라이언트 실행
`python client.py`
- 각 클라이언트는 서로 다른 PC 또는 Colab에서 실행 가능
- 실행 시 서버에 자동 접속


## 7. 학습 흐름 요약
1. 서버 실행 후 대기
2. 클라이언트 접속
3. Dirichlet α 기반 Non-IID 데이터 분할
4. 로컬 학습(U-Net + Dice Loss)
5. 모델 파라미터 서버 전송
6. 서버에서 FedAvg 기반 집계
7. 설정된 라운드 수만큼 반복 후 종료


## 8. 출력 결과
- 콘솔 출력
  - 라운드 별 Training Loss
  - Validation Dice Score
- 학습 종료 메시지  
```
[SUMMARY] Round-wise results
Training Completed. Server Terminated.
```


## 9. 종료 방법
- 자동 종료(설정된 라운드 수 완료 시)
- 수동 종료 시 `ctrl + c`


## 10. 주의 사항 및 제한점
- 논문 원본 코드 미제공 → 구조 기반 재구현
- FedFM 핵심(SCAFFOLD 제어변수 + Feature Mean 보정)은 부분 반영
- GPU 사용 시 PyTorch CUDA 버전 호환 필요
- 데이터 파일명 불일치 시 로딩 실패 가능
