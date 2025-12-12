import flwr as fl
import random
import numpy as np
import torch


#  실험 재현성을 위한 Seed 고정
def set_seed(seed: int = 42):
    """
    랜덤 시드를 고정하여 Federated Learning 실험의 재현성을 보장함.
    클라이언트 수, 데이터셋 샘플링, weight 초기화 등에서 결과 변동 최소화.

    - random : 기본 파이썬 난수
    - numpy  : 연산 및 weight 초기화
    - torch  : PyTorch 모델 사용 시 동일 seed 보장
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)



#  서버 주소 설정
# 서버가 바인딩할 주소 (0.0.0.0 → 외부에서 접속 가능)
# Flower는 gRPC 기반이므로 host:port 형태로 작성해야 함
SERVER_ADDRESS = "0.0.0.0:8080"



# 라운드별 클라이언트 설정 전달 함수
def fit_config(round_number: int):
    """
    각 Round가 시작될 때 클라이언트에게 전달할 Configuration을 정의.
    논문 실험에서 FedAvg의 주요 하이퍼파라미터를 조절할 수 있음.

    파라미터:
        round_number(int): 현재 Federated Round 번호

    반환: dict
        - local_epochs : 클라이언트 로컬 모델 학습 epoch 수
        - batch_size   : 클라이언트에서 사용할 미니배치 크기
        - learning_rate: 로컬 학습 learning rate
        - round        : 로깅을 위해 Round 번호 전달
    """
    return {
        "local_epochs": 1,        # 클라이언트가 Round마다 수행할 로컬 학습 epoch 수
        "batch_size": 32,         # 논문 실험에서 설정한 기본값
        "learning_rate": 0.001,   # FedAvg 기본 learning rate
        "round": round_number     # 클라이언트에서 로깅용으로 사용
    }



# Federated Averaging 전략 정의

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,            # 매 라운드에 클라이언트 전체 중 참여 비율 (1.0 → 모두 참여)
    min_fit_clients=3,           # federated 학습을 진행하기 위해 필요한 최소 학습 클라이언트 수
    min_available_clients=3,     # 서버가 학습을 시작하기 위해 필요한 최소 접속 클라이언트 수
    on_fit_config_fn=fit_config, # 매 Round 시작 시 클라이언트에게 config 전달
)


#  서버 구동 함수

def main():
    """
    Federated Learning 서버를 실행하는 메인 함수.
    FedAvg 전략을 기반으로 5개의 Round 진행.
    """

    # 서버 시작 로깅
    print("\n" + "=" * 70)
    print(" [FEDERATED SERVER] Flower FedAvg Server")
    print(f" Address   : {SERVER_ADDRESS}")    # 서버 주소 출력
    print(" Strategy  : FedAvg")               # FedAvg 전략 사용
    print(" Rounds : 5 (Phase 2 기준 실험값)")  # 실험 계획에 따라 Round 수 설정
    print("=" * 70 + "\n")

    # 서버 실행
    # Flower의 gRPC 기반 플라워 서버 시작
    # - num_rounds: Federated Learning round 반복 횟수
    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

    # 학습 종료 메시지 출력
    print("\n" + "=" * 70)
    print(" [SERVER] Training Completed. Server Terminated.")
    print("=" * 70)



if __name__ == "__main__":
    # python server.py 실행 시 main() 호출
    main()
