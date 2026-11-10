"""롤 10분 시점 스탯으로 블루팀 승패를 예측하는 LightGBM 학습 스크립트.

데이터: Kaggle "League of Legends Diamond Ranked Games (10 min)"
        파일명 high_diamond_ranked_10min.csv  ->  data/ 폴더에 넣기

실행: python train.py
결과: model.txt(모델), importance.csv(변수 중요도), 콘솔에 정확도/AUC/속도 비교
"""
import time
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from features import FEATURES, TARGET

DATA = Path("data/high_diamond_ranked_10min.csv")

if not DATA.exists():
    raise SystemExit(
        f"{DATA} 파일이 없어요. Kaggle에서 'League of Legends Diamond Ranked Games (10 min)'를 "
        "받아서 data 폴더에 넣어주세요."
    )

df = pd.read_csv(DATA)
X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"데이터 {len(df):,}판 / 학습 {len(X_train):,} / 테스트 {len(X_test):,}")

# 1) 실제 서비스용 모델: 조기 종료로 적정 트리 수를 자동으로 찾음
model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=15,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)
model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)
proba = model.predict_proba(X_test)[:, 1]
print(f"[LightGBM] 정확도 {accuracy_score(y_test, proba > 0.5):.4f} / AUC {roc_auc_score(y_test, proba):.4f}")

model.booster_.save_model("model.txt")
imp = pd.Series(model.booster_.feature_importance("gain"), index=FEATURES)
imp = (imp / imp.sum()).sort_values(ascending=False)
imp.to_csv("importance.csv", header=["importance"])
print("\n변수 중요도(상위 5):")
print(imp.head(5).round(3).to_string())

# 2) 속도 비교: 같은 트리 수(200)로 LightGBM vs sklearn GradientBoosting
N = 200
t0 = time.perf_counter()
lgb.LGBMClassifier(n_estimators=N, learning_rate=0.1, verbose=-1, random_state=42).fit(X_train, y_train)
t_lgb = time.perf_counter() - t0

t0 = time.perf_counter()
GradientBoostingClassifier(n_estimators=N, learning_rate=0.1, random_state=42).fit(X_train, y_train)
t_sk = time.perf_counter() - t0

print(f"\n[속도 비교, 트리 {N}개] LightGBM {t_lgb:.2f}초 / sklearn GradientBoosting {t_sk:.2f}초 ({t_sk / t_lgb:.1f}배)")
