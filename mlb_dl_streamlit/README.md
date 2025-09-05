# MLB Game Outcome Predictor (python-mlb-statsapi + PyTorch + Streamlit)

이 예제 프로젝트는 [`python-mlb-statsapi`](https://pypi.org/project/python-mlb-statsapi/)로 MLB 데이터를 수집하고,
간단한 딥러닝(MLP) 모델로 홈팀 승리 확률을 예측한 뒤, Streamlit 대시보드로 예측값을 확인하는 **기본 구조**입니다.

> ⚠️ 실제 베팅/상업적 용도 금지. 교육/연구 목적의 최소 구현 예제입니다.

## 구성
```text
mlb_dl_streamlit/
├─ app.py                 # Streamlit 앱 (예측 UI)
├─ data_fetch.py          # 일정/팀 스탯 수집
├─ features.py            # 피처 엔지니어링
├─ train.py               # 학습 스크립트 (PyTorch MLP)
├─ utils.py               # 공용 유틸 (MLB 클라이언트 등)
├─ models/
│   └─ model.pt           # 학습된 모델 (학습 후 생성)
├─ data/
│   ├─ games_YYYYMMDD_YYYYMMDD.parquet  # 일정/결과 캐시
│   └─ team_stats_SEASON.parquet        # 팀 스탯 캐시
└─ requirements.txt
```

## 빠른 시작
```bash
# 1) 환경 구성
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) 과거 데이터로 간단 학습 (예: 2024 시즌 04-01 ~ 08-31)
python train.py --start 2024-04-01 --end 2024-08-31 --season 2024

# 3) Streamlit 실행 (오늘 경기 예측)
streamlit run app.py
```

## 아이디어
- 현재는 **시즌 누적 팀 스탯 차이(홈-원정)** + **최근 10경기 승률** 등 아주 단순한 피처만 사용합니다.
- 실전에서는 선발 투수, 라인업, 부상, 원정 이동거리, 날씨, 구장 특성 등 더 풍부한 피처를 추가하세요.
- 모델도 심층 네트워크/시계열(RNN/Transformer), 캘리브레이션 등으로 고도화할 수 있습니다.
