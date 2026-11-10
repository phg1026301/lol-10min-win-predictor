# 학습(train.py)과 웹(app.py)이 같은 입력 순서를 쓰도록 한 곳에서 관리합니다.
# 모든 값은 "블루팀 기준, 게임 시작 10분 시점" 데이터입니다.
FEATURES = [
    "blueKills",
    "blueDeaths",
    "blueAssists",
    "blueFirstBlood",
    "blueDragons",
    "redDragons",
    "blueHeralds",
    "redHeralds",
    "blueTowersDestroyed",
    "redTowersDestroyed",
    "blueTotalMinionsKilled",
    "redTotalMinionsKilled",
    "blueGoldDiff",
    "blueExperienceDiff",
]
TARGET = "blueWins"
