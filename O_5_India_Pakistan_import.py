# =======================================
# 0. 환경 준비
# =======================================
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.oauth2 import service_account

# =======================================
# 1. Google Cloud 인증
# =======================================
KEY_PATH = "C:\\AIproject\\my-gdelt-analysis-2025-8990d16f9b7a.json"
credentials = service_account.Credentials.from_service_account_file(KEY_PATH)

# =======================================
# 2. BigQuery 클라이언트 설정
# =======================================
PROJECT_ID = "my-gdelt-analysis-2025"
client = bigquery.Client(project=PROJECT_ID, credentials=credentials)

# =======================================
# 3. GDELT 데이터 쿼리 (인도-파키스탄, 2025년)
# =======================================
# '이상 징후' 기간(4-5월)과 그 이전의 '정상' 기간을 모두 포함
AGGREGATED_QUERY = """
WITH RelevantEvents AS (
  SELECT
    SQLDATE,
    EventRootCode
  FROM
    `gdelt-bq.gdeltv2.events`
  WHERE
    -- 기간: 2025년 1월 1일 ~ 2025년 7월 31일
    SQLDATE BETWEEN 20250101 AND 20250731
    -- 국가: 인도(IND) 및 파키스탄(PAK)
    AND (
      Actor1CountryCode IN ('IND', 'PAK') OR
      Actor2CountryCode IN ('IND', 'PAK') OR
      ActionGeo_CountryCode IN ('IND', 'PAK')
    )
)
-- EventRootCode만 집계
SELECT
  SQLDATE,
  EventRootCode,
  COUNT(*) as event_count
FROM RelevantEvents
GROUP BY SQLDATE, EventRootCode
"""

print("[INFO] BigQuery에서 데이터를 불러오는 중입니다 (인도/파키스탄 2025)...")
# BigQuery Storage API 관련 경고가 나올 수 있으나, 실행에 문제 없습니다.
df_aggregated = client.query(AGGREGATED_QUERY).to_dataframe()
print("[INFO] 데이터 로딩 완료. Shape:", df_aggregated.shape)

# =======================================
# 4. 데이터 처리 및 비율 계산 (이전 오류 수정됨)
# =======================================
df_aggregated['date'] = pd.to_datetime(df_aggregated['SQLDATE'], format='%Y%m%d')

df_root = df_aggregated
# [수정 완료] 'EventRootCode' 컬럼을 읽어서 'code'라는 새 컬럼을 생성합니다.
df_root['code'] = df_root['EventRootCode'].astype(str).str.zfill(2) 
root_dist = df_root.pivot(index='date', columns='code', values='event_count').fillna(0)
root_ratio = root_dist.div(root_dist.sum(axis=1), axis=0).fillna(0)

# EventRootCode 01~20까지 모든 컬럼을 확보 (없는 코드는 0으로 채움)
all_codes = [str(i).zfill(2) for i in range(1, 21)]
for code in all_codes:
    if code not in root_ratio.columns:
        root_ratio[code] = 0.0
# 컬럼 순서 정렬
root_ratio = root_ratio[all_codes]

print("[INFO] 데이터 비율 계산 완료. Shape:", root_ratio.shape)

# =======================================
# 5. 레이블링 (Labeling)
# =======================================
# a. 이벤트 날짜 및 윈도우 정의
EVENT_DATE = pd.to_datetime('2025-05-07') # 인도의 '신두르 작전' 공습 시작일
WINDOW_DAYS = 30  # 분쟁 직전 30일을 '이상 징후'로 정의
ANOMALY_START_DATE = EVENT_DATE - pd.Timedelta(days=WINDOW_DAYS)
# ANOMALY_START_DATE = '2025-04-07'
# (참고: 4/22 파할감 테러가 이 '이상 징후' 기간 내에 포함됩니다)

# b. 'is_anomaly' 컬럼을 0으로 초기화
root_ratio['is_anomaly'] = 0

# c. '이상 징후' 기간 (4/7 ~ 5/6)에 해당하는 행의 라벨을 1로 변경
mask = (root_ratio.index >= ANOMALY_START_DATE) & (root_ratio.index < EVENT_DATE)
root_ratio.loc[mask, 'is_anomaly'] = 1

# d. 이벤트 발생 당일(5/7) 및 그 이후 데이터는 학습에서 제외
df_labeled_ind_pak_2025 = root_ratio[root_ratio.index < EVENT_DATE].copy()

# =======================================
# 6. 결과 확인 및 저장
# =======================================
print("\n[SUCCESS] 인도-파키스탄 (2025) 레이블링 완료.")
print(f"최종 데이터 Shape: {df_labeled_ind_pak_2025.shape}")

print("\n--- '이상 징후' 시작 지점 데이터 (Label=1) ---")
print(df_labeled_ind_pak_2025[df_labeled_ind_pak_2025['is_anomaly'] == 1].head(3))

print("\n--- '정상' 기간 데이터 (Label=0) ---")
print(df_labeled_ind_pak_2025[df_labeled_ind_pak_2025['is_anomaly'] == 0].head(3))

# 이 데이터프레임을 나중에 다른 사례와 합치기 위해 파일로 저장합니다.
df_labeled_ind_pak_2025.to_csv("labeled_ind_pak_2025.csv")
print("\n[INFO] 'labeled_ind_pak_2025.csv' 파일로 저장 완료.")