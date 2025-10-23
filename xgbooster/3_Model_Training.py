import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import classification_report, precision_recall_curve, auc

# =======================================
# 1. 파라미터 및 파일 설정
# =======================================
# 2단계에서 생성된 피처 데이터셋 파일
INPUT_FILE = "features_dataset_30d.csv" 

# 시계열 데이터를 나눌 비율 (예: 80% 학습, 20% 테스트)
TEST_SPLIT_RATIO = 0.2

# =======================================
# [🚨 디버깅 코드 추가 🚨]
# =======================================
print("[DEBUG] 2단계 파일의 'is_anomaly' 컬럼 값을 검사합니다...")
if not os.path.exists(INPUT_FILE):
    print(f"[ERROR] '{INPUT_FILE}'을 찾을 수 없습니다.")
    sys.exit()

# 파일을 임시로 불러와서 'is_anomaly' 컬럼만 검사
try:
    temp_df = pd.read_csv(INPUT_FILE)
    
    # 'is_anomaly' 컬럼에 있는 고유한(unique) 값들을 모두 찾습니다.
    unique_values = temp_df['is_anomaly'].unique()
    
    print(f"       [DEBUG] 'is_anomaly' 컬럼의 고유 값: {unique_values}")
    
    # 0과 1 외의 값이 있는지 확인
    non_binary_values = [v for v in unique_values if v not in [0, 1]]
    
    if len(non_binary_values) > 0:
        print("\n[FATAL ERROR] 2단계 파일에 0과 1 외의 값이 포함되어 있습니다!")
        print(f"       비정상 값: {non_binary_values}")
        print("       이 '삼각형' 데이터가 그래프 문제를 일으키는 원인입니다.")
        print("       1단계(unify) 스크립트부터 다시 실행하여 데이터를 정제하세요.")
        sys.exit() # 0과 1 외의 값이 있으면 스크립트 중지
    else:
        print("       [DEBUG] 좋습니다. 'is_anomaly' 컬럼에 0과 1만 존재합니다.")
        
except Exception as e:
    print(f"[ERROR] 디버깅 중 파일({INPUT_FILE})을 읽는 데 실패했습니다: {e}")
    sys.exit()

del temp_df # 검사 후 메모리에서 삭제
print("="*50 + "\n")
# =======================================
# [ 🚨 디버깅 코드 끝 🚨 ]
# =======================================


# =======================================
# 2. 피처 데이터 불러오기
# =======================================
print(f"[INFO] 2단계에서 생성된 피처 데이터 '{INPUT_FILE}'를 불러옵니다...")
if not os.path.exists(INPUT_FILE):
    print(f"[ERROR] '{INPUT_FILE}'을 찾을 수 없습니다.")
    print("       먼저 2단계(feature_engineering) 스크립트를 실행했는지 확인하세요.")
    sys.exit()

# 'date' 컬럼을 인덱스로, 날짜 형식으로 불러옵니다.
df = pd.read_csv(INPUT_FILE, index_col='date', parse_dates=True)
print(f"       데이터 불러오기 완료. Shape: {df.shape}")

# =======================================
# 3. 학습(X) / 정답(y) 분리
# =======================================
print("[INFO] 데이터를 피처(X)와 레이블(y)로 분리합니다...")
y = df['is_anomaly']
X = df.drop(columns=['is_anomaly'])

# =======================================
# 4. (중요) 시계열 학습/테스트 데이터 분리
# =======================================
# 🚨 경고: 시계열 데이터는 절대 섞으면 안 됩니다 (shuffle=False).
# 과거 데이터로 학습(Train)하고, 미래 데이터로 검증(Test)해야 합니다.


split_index = int(len(X) * (1 - TEST_SPLIT_RATIO))

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print(f"       학습 데이터(Train): {X_train.shape[0]}일치")
print(f"       테스트 데이터(Test): {X_test.shape[0]}일치")
print(f"       테스트 시작 날짜: {X_test.index.min().date()}")

# =======================================
# 5. 모델 정의 및 학습 (XGBoost)
# =======================================
# 💡 (핵심) 데이터 불균형 처리 (Imbalanced Data)
# '정상(0)' 샘플이 '이상(1)' 샘플보다 훨씬 많습니다.
# 'scale_pos_weight'는 '이상(1)' 클래스에 얼마나 더 많은 가중치(패널티)를 
# 줄지 계산합니다. (정상 개수 / 이상 개수)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\n[INFO] 불균형 데이터 가중치 (scale_pos_weight): {scale_pos_weight:.2f}")

# XGBoost 분류기 모델을 정의합니다.
model = XGBClassifier(
    n_estimators=100,      # 트리의 개수 (튜닝 가능)
    learning_rate=0.1,     # 학습률 (튜닝 가능)
    scale_pos_weight=scale_pos_weight, # 💡 불균형 처리를 위한 핵심 파라미터
    random_state=42,       # 재현성을 위한 시드
    n_jobs=-1              # 모든 CPU 코어 사용
)

print("\n[INFO] XGBoost 모델 학습을 시작합니다...")
model.fit(X_train, y_train)
print("[INFO] 모델 학습 완료.")

# =======================================
# [🚨 4단계를 위해 이 부분을 추가하세요! 🚨]
# =======================================
MODEL_FILE = "xgb_model.json"
print(f"[INFO] 4단계를 위해 학습된 모델을 '{MODEL_FILE}'에 저장합니다...")
model.save_model(MODEL_FILE)
print("[INFO] 모델 저장 완료.")
# =======================================
# [ 🚨 추가 끝 🚨 ]
# =======================================


# =======================================
# 6. 모델 평가 (분류 리포트)
# =======================================
# 0.5 임계값 기준으로 0 또는 1 예측
y_pred = model.predict(X_test)

print("\n" + "="*50)
print(" [ 모델 성능 평가: 분류 리포트 ]")
print("="*50)
# '1' (Anomaly) 클래스의 Precision(정밀도)과 Recall(재현율)이 중요합니다.
# Precision: 모델이 "이상"이라고 한 것 중, 진짜 "이상"인 비율 (거짓 경보)
# Recall:    실제 "이상" 중에서, 모델이 "이상"이라고 맞춘 비율 (놓친 경보)
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Anomaly (1)']))
print("="*50 + "\n")


# =======================================
# 7. (중요) 위험 확률 예측 및 시각화
# =======================================
# 사용자님이 요청하신 '위험 확률'을 계산합니다.
# .predict_proba()는 [P(0), P(1)]을 반환하므로, '이상 징후일 확률' P(1)을 [:, 1]로 선택
probabilities = model.predict_proba(X_test)[:, 1]

# 결과를 시각화하기 편하게 DataFrame으로 만듭니다.
df_results = pd.DataFrame({
    'Actual_Anomaly': y_test,
    'Predicted_Probability': probabilities
}, index=y_test.index)

# [수정] 'Actual_Anomaly'를 정수로 강제 변환
df_results['Actual_Anomaly'] = df_results['Actual_Anomaly'].astype(int) 

print("[INFO] '위험 확률' 예측 및 시각화를 시작합니다...")

# [수정] VS Code 플롯 캐시 클리어
plt.clf()
plt.close('all')

fig, ax1 = plt.subplots(figsize=(16, 6))

# 1. (파란색) 예측된 위험 확률 플로팅
color = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('Predicted Risk Probability', color=color)
# 'label'을 추가
ax1.plot(df_results.index, df_results['Predicted_Probability'], color=color, label='Risk Probability')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, 1) # 확률은 0에서 1 사이

# 2. (빨간색) 실제 이상 징후(정답) 플로팅
color = 'tab:red'
ax2 = ax1.twinx() 
ax2.set_ylabel('Actual Anomaly (1=True)', color=color)

# [🚨 핵심 수정 🚨]
# fill_between 대신 'plot'을 사용하고 'drawstyle'을 'steps-post'로 설정합니다.
# 이렇게 하면 0과 1 사이를 '선'이 아닌 '계단'으로 그려서 '삼각형' 현상을 없앱니다.
ax2.plot(df_results.index, df_results['Actual_Anomaly'], 
         color=color, alpha=0.5, label='Actual Anomaly Period', 
         drawstyle='steps-post') # <-- 이 부분이 '계단'으로 그리게 만듭니다.

ax2.tick_params(axis='y', labelcolor=color)
# Y축을 0-5가 아닌 0과 1에 맞게 조정합니다 (선의 최소/최대값)
ax2.set_ylim(-0.1, 1.1) 

plt.title('Anomaly Risk Prediction vs Actual Events (Test Set)')
fig.tight_layout()

# [수정] 범례(Legend)가 올바르게 나오도록 수정
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')

plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# =======================================
# 8. (보너스) 피처 중요도 시각화
# =======================================
# 모델이 어떤 피처(예: '19_mean_30d')를 중요하게 봤는지 확인합니다.

print("[INFO] 모델이 중요하게 생각하는 Top 15 피처를 시각화합니다...")
fig, ax = plt.subplots(figsize=(10, 8))
plot_importance(model, ax=ax, max_num_features=15, importance_type='gain')
plt.title('Feature Importance (based on Gain)')
plt.show()