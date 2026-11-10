---
title: LoL 10min Win Predictor
emoji: 🎮
sdk: gradio
app_file: app.py
---

# 롤 10분 승률 예측기 (LightGBM)

Kaggle "League of Legends Diamond Ranked Games (10 min)" 데이터로 LightGBM을 학습해서,
10분 시점 스탯으로 블루팀 승률을 보여주는 웹 앱입니다.

## 실행 순서
1. Kaggle에서 `high_diamond_ranked_10min.csv`를 받아 `data/` 폴더에 넣기
2. `pip install -r requirements.txt`
3. `python train.py`  → `model.txt`, `importance.csv` 생성, 속도 비교 출력
4. `python app.py`    → 로컬에서 웹 앱 확인

## 배포 (Hugging Face Spaces)
Space를 Gradio SDK로 만들고 `app.py`, `features.py`, `model.txt`, `requirements.txt`, `README.md`를 올리면 끝.
