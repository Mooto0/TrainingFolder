import pandas as pd
import numpy as np
import sys
import os

# =======================================
# 1. 파라미터 설정
# =======================================
# 🚨 여기가 가장 중요합니다! 
# 이 윈도우 크기를 30, 45, 60 등으로 바꿔가며
# 모델의 성능이 어떻게 변하는지 실험(튜닝)하게 됩니다.
WINDOW_SIZE = 14  

INPUT_FILE = "unified_dataset.csv"
OUTPUT_FILE = f"features_dataset_{WINDOW_SIZE}d.csv"

# =======================================
# 2. 통합 데이터 불러오기
# =======================================
print(f"[INFO] 1단계에서 통합된 데이터 '{INPUT_FILE}'를 불러옵니다...")
if not os.path.exists(INPUT_FILE):
    print(f"[ERROR] '{INPUT_FILE}'을 찾을 수 없습니다.")
    print("       먼저 1단계(unify) 스크립트를 실행했는지 확인하세요.")
    sys.exit()

# 'date' 컬럼을 인덱스로, 날짜 형식으로 불러옵니다.
final_dataset = pd.read_csv(INPUT_FILE, index_col='date', parse_dates=True)
print(f"       데이터 불러오기 완료. Shape: {final_dataset.shape}")

# =======================================
# 3. 피처 엔지니어링 (슬라이딩 윈도우)
# =======================================
# 

print(f"[INFO] {WINDOW_SIZE}일 롤링 윈도우 피처 엔지니어링을 시작합니다...")

# 01~20 EventRootCode 컬럼 목록
features = [str(i).zfill(2) for i in range(1, 21)]

# 생성된 피처를 담을 새 데이터프레임
df_features = pd.DataFrame(index=final_dataset.index)

# 20개 코드에 대해 롤링(Rolling) 통계 계산
for col in features:
    if col in final_dataset.columns:
        # 30일 이동 평균 (최근 30일의 평균적인 값)
        df_features[f'{col}_mean_{WINDOW_SIZE}d'] = final_dataset[col].rolling(window=WINDOW_SIZE).mean()
        
        # 30일 이동 표준편차 (최근 30일의 변동성)
        df_features[f'{col}_std_{WINDOW_SIZE}d'] = final_dataset[col].rolling(window=WINDOW_SIZE).std()
    else:
        print(f"[WARNING] '{col}' 컬럼이 원본 데이터에 없습니다. 건너뜁니다.")

print("       롤링 통계 계산 완료.")

# =======================================
# 4. 레이블(정답) 복사
# =======================================
# 'is_anomaly' (정답) 컬럼은 피처가 아니므로 그대로 복사합니다.
df_features['is_anomaly'] = final_dataset['is_anomaly']

# =======================================
# 5. 결측치(NaN) 제거
# =======================================
# 롤링 윈도우 특성상, 데이터의 맨 처음 (WINDOW_SIZE - 1)일 (즉, 29일)
# 동안은 통계 계산이 불가능하므로 NaN(결측치)이 발생합니다.
# 이 값들은 모델 학습에 사용할 수 없으므로 반드시 제거해야 합니다.
print(f"\n[INFO] 결측치(NaN) 제거 전 Shape: {df_features.shape}")
df_model_ready = df_features.dropna()
print(f"       결측치(NaN) 제거 후 Shape: {df_model_ready.shape}")

# =======================================
# 6. 최종 파일로 저장
# =======================================
# 3단계(모델 학습)에서 사용할 최종 데이터셋을 저장합니다.
print(f"\n[INFO] 피처 엔지니어링 완료된 데이터를 '{OUTPUT_FILE}'로 저장합니다...")
df_model_ready.to_csv(OUTPUT_FILE, index=True)

print(f"[SUCCESS] 저장 완료! '{OUTPUT_FILE}'가 생성되었습니다.")
print("       이제 3단계(모델 학습)를 진행할 수 있습니다.")

print("\n--- 최종 생성된 데이터 샘플 (상위 5개) ---")
# 20개 코드 * 2개 통계 = 40개 피처 + 1개 레이블 = 41개 컬럼
print(df_model_ready.head())