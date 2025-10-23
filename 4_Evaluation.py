import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import json
from xgboost import XGBClassifier

# =======================================
# 1. 파일 이름 정의
# =======================================
# 3단계에서 저장한 모델
MODEL_FILE = "xgb_model.json" 
# 2단계에서 생성한 (학습+테스트가 합쳐진) 전체 피처 데이터
FEATURE_DATA_FILE = "features_dataset_30d.csv" 
# 4단계 최종 결과물을 저장할 파일
OUTPUT_FILE = "historical_risk_profile.csv"

# =======================================
# 2. 모델 및 전체 피처 데이터 로드
# =======================================
print(f"[INFO] 3단계에서 저장한 '{MODEL_FILE}'에서 모델을 로드합니다...")

# 파일들이 있는지 확인
if not all(os.path.exists(f) for f in [MODEL_FILE, FEATURE_DATA_FILE]):
    print("[ERROR] 필수 파일(model 또는 feature data)이 없습니다.")
    print(f"       '{MODEL_FILE}' 또는 '{FEATURE_DATA_FILE}' 파일이 있는지 확인하세요.")
    print("       3단계 스크립트에 모델 저장 코드를 추가하고 실행했는지 확인하세요.")
    sys.exit()

# 빈 XGBoost 모델을 만들고, 저장된 가중치를 불러옵니다.
model = XGBClassifier()
model.load_model(MODEL_FILE)
print("[INFO] 모델 로드 완료.")

print(f"[INFO] 2단계의 '{FEATURE_DATA_FILE}'에서 전체 피처 데이터를 로드합니다...")
df = pd.read_csv(FEATURE_DATA_FILE, index_col='date', parse_dates=True)

# 레이블(y)과 피처(X) 분리
y_all_actual = df['is_anomaly']
# 3단계에서 학습할 때 'is_anomaly'를 제외했으므로, 적용할 때도 동일하게 제외
X_all = df.drop(columns=['is_anomaly'])
print(f"       전체 데이터 로드 완료. Shape: {X_all.shape}")

# =======================================
# 3. 전체 데이터에 '위험 확률' 예측 적용
# =======================================
print("[INFO] 전체 기간에 대해 '위험 확률'을 예측합니다...")

# 모델을 사용하여 P(1), 즉 '이상 징후일 확률'을 계산합니다.
all_probabilities = model.predict_proba(X_all)[:, 1]

# 결과를 원본 정답(Actual)과 함께 DataFrame으로 결합
df_results = pd.DataFrame({
    'Actual_Anomaly': y_all_actual,
    'Predicted_Probability': all_probabilities
}, index=df.index)

print("[INFO] 예측 완료.")

# =======================================
# 4. 최종 결과 파일로 저장
# =======================================
print(f"[INFO] 전체 위험도 프로필을 '{OUTPUT_FILE}'로 저장합니다...")
df_results.to_csv(OUTPUT_FILE)
print("[INFO] 저장 완료.")

# =======================================
# 5. 전체 기간 시각화 (최종 결과물)
# =======================================
# 이 그래프는 3단계의 'Test Set' 그래프와 달리,
# 모델이 학습했던 기간(과거)과 테스트했던 기간(미래)을 모두 포함합니다.


print("[INFO] 전체 기간에 대한 'Historical Risk Profile' 그래프를 생성합니다...")
fig, ax1 = plt.subplots(figsize=(16, 6))

# 1. (파란색) 예측된 위험 확률 플로팅
color = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('Predicted Risk Probability (0.0 to 1.0)', color=color)
ax1.plot(df_results.index, df_results['Predicted_Probability'], 
         color=color, label='Risk Probability', alpha=0.8, linewidth=1.5)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, 1)

# 2. (빨간색) 실제 이상 징후(정답) 플로팅
color = 'tab:red'
ax2 = ax1.twinx() # x축을 공유하는 두 번째 y축
ax2.set_ylabel('Actual Anomaly (1=True)', color=color)
# fill_between으로 빨간색 '이상 징후 기간'을 음영 처리
ax2.fill_between(df_results.index, 0, df_results['Actual_Anomaly'], 
                 color=color, alpha=0.3, label='Actual Anomaly Period')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 5) # 빨간색 영역이 잘 보이도록 Y축 스케일 조정

plt.title('Historical Anomaly Risk Profile (All Data)')
fig.tight_layout()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()