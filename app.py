"""학습된 model.txt를 불러와 승률을 보여주는 Gradio 웹 앱."""
import gradio as gr
import lightgbm as lgb
import pandas as pd

from features import FEATURES

model = lgb.Booster(model_file="model.txt")

# 변수명 -> (화면 라벨, 최소, 최대, 기본값, 단계)
SPEC = {
    "blueKills": ("블루 킬", 0, 30, 6, 1),
    "blueDeaths": ("블루 데스(레드 킬)", 0, 30, 6, 1),
    "blueAssists": ("블루 어시스트", 0, 50, 7, 1),
    "blueDragons": ("블루 드래곤", 0, 1, 0, 1),
    "redDragons": ("레드 드래곤", 0, 1, 0, 1),
    "blueHeralds": ("블루 전령", 0, 1, 0, 1),
    "redHeralds": ("레드 전령", 0, 1, 0, 1),
    "blueTowersDestroyed": ("블루 타워 파괴", 0, 4, 0, 1),
    "redTowersDestroyed": ("레드 타워 파괴", 0, 4, 0, 1),
    "blueTotalMinionsKilled": ("블루 CS(미니언)", 0, 400, 217, 1),
    "redTotalMinionsKilled": ("레드 CS(미니언)", 0, 400, 217, 1),
    "blueGoldDiff": ("골드 차이(블루-레드)", -10000, 10000, 0, 50),
    "blueExperienceDiff": ("경험치 차이(블루-레드)", -8000, 8000, 0, 50),
}


def predict(first_blood, *values):
    row = dict(zip([f for f in FEATURES if f != "blueFirstBlood"], values))
    row["blueFirstBlood"] = int(first_blood)
    p = float(model.predict(pd.DataFrame([row])[FEATURES])[0])
    return {"블루팀 승리": p, "레드팀 승리": 1 - p}


inputs = [gr.Checkbox(label="블루 퍼스트 블러드", value=False)]
for name in FEATURES:
    if name == "blueFirstBlood":
        continue
    label, lo, hi, default, step = SPEC[name]
    inputs.append(gr.Slider(lo, hi, value=default, step=step, label=label))

demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=gr.Label(label="10분 시점 예상 승률"),
    title="롤 10분 승률 예측기 (LightGBM)",
    description="게임 시작 10분 시점의 스탯을 입력하면 블루팀 승률을 예측해요.\n\n⚠️ 다이아몬드 이상 랭크게임 데이터로 학습했어요. 다른 티어에서는 정확도가 떨어질 수 있어요.",
    live=True,
)

if __name__ == "__main__":
    import os

    # Render 등 배포 환경은 PORT 환경변수로 포트를 알려줍니다.
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
