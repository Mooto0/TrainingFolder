# =======================================
# 0. 환경 준비
# =======================================
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.oauth2 import service_account
import sys # [NEW] 스크립트 종료를 위해 추가

# =======================================
# [NEW] 헬퍼 함수: 바이트를 읽기 쉽게 변환
# =======================================
def format_bytes(bytes_val):
    """바이트 값을 KB, MB, GB, TB 단위로 변환합니다."""
    if bytes_val is None:
        return "N/A"
    if bytes_val < 1024**2:
        return f"{bytes_val / 1024:.2f} KB"
    elif bytes_val < 1024**3:
        return f"{bytes_val / 1024**2:.2f} MB"
    elif bytes_val < 1024**4:
        return f"{bytes_val / 1024**3:.2f} GB"
    else:
        return f"{bytes_val / 1024**4:.2f} TB"

# =======================================
# 1. Google Cloud 인증
# =======================================
KEY_PATH = "C:\\AIproject\\my-bigquery-project-2025-3b545e4abbc6.json"
credentials = service_account.Credentials.from_service_account_file(KEY_PATH)

# =======================================
# 2. BigQuery 클라이언트 설정
# =======================================
PROJECT_ID = "my-bigquery-project-2025"
client = bigquery.Client(project=PROJECT_ID, credentials=credentials)

# =======================================
# 3. GDELT 데이터 쿼리 (아르메니아-아제르바이잔, 2020년)
# =======================================
# '이상 징후' 기간과 그 이전의 '정상' 기간을 모두 포함
AGGREGATED_QUERY = """
WITH RelevantEvents AS (
  SELECT
    SQLDATE,
    EventRootCode
  FROM
    `gdelt-bq.gdeltv2.events`
  WHERE
    -- 기간: 2020년 6월 1일 ~ 2020년 12월 31일
    SQLDATE BETWEEN 20200601 AND 20201231
    -- 국가: 아르메니아(ARM) 및 아제르바이잔(AZE)
    AND (
      Actor1CountryCode IN ('ARM', 'AZE') OR
      Actor2CountryCode IN ('ARM', 'AZE') OR
      ActionGeo_CountryCode IN ('ARM', 'AZE')
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

# =======================================
# [NEW] 3.5. 쿼리 실행 전 건조 실행(Dry Run) 및 용량 확인
# =======================================
# 1. 건조 실행(Dry Run)을 위한 쿼리 작업 설정
job_config = bigquery.QueryJobConfig(
    dry_run=True,          # 실제 쿼리를 실행하지 않음
    use_query_cache=False, # 캐시된 결과를 사용하지 않아 정확한 스캔량 예측
)

print("[INFO] 쿼리 예상 사용량을 확인하기 위해 '건조 실행(Dry Run)'을 시작합니다...")

try:
    # 2. 건조 실행 쿼리 전송
    dry_run_job = client.query(AGGREGATED_QUERY, job_config=job_config)
    
    # 3. 예상 스캔 용량(바이트) 확인
    estimated_bytes = dry_run_job.total_bytes_processed
    estimated_cost_str = format_bytes(estimated_bytes)
    
    print(f"\n[INFO] 예상 스캔 용량: {estimated_cost_str} ({estimated_bytes} bytes)")
    
    # 4. 사용자에게 실행 여부 확인
    print("       (참고: BigQuery는 매월 1TB의 무료 처리 용량을 제공합니다.)")
    proceed = input(f"       쿼리를 계속 실행하시겠습니까? [y/n]: ")

    if proceed.lower() != 'y':
        print("[INFO] 사용자가 쿼리 실행을 취소했습니다.")
        sys.exit() # 스크립트 종료

except Exception as e:
    print(f"[ERROR] 쿼리 검증 또는 건조 실행 중 오류 발생: {e}")
    sys.exit()

# =======================================
# [MOVED] 3.8. 실제 쿼리 실행 (사용자가 'y'를 입력한 경우)
# =======================================
print("\n[INFO] BigQuery에서 실제 데이터를 불러오는 중입니다 (아르메니아/아제르바이잔)...")
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
EVENT_DATE = pd.to_datetime('2020-09-07') # 사용자 요청 기준일
WINDOW_DAYS = 30  # 기준일 직전 30일을 '이상 징후'로 정의
ANOMALY_START_DATE = EVENT_DATE - pd.Timedelta(days=WINDOW_DAYS)
# ANOMALY_START_DATE = '2020-08-08'

# b. 'is_anomaly' 컬럼을 0으로 초기화
root_ratio['is_anomaly'] = 0

# c. '이상 징후' 기간 (8/8 ~ 9/6)에 해당하는 행의 라벨을 1로 변경
mask = (root_ratio.index >= ANOMALY_START_DATE) & (root_ratio.index < EVENT_DATE)
root_ratio.loc[mask, 'is_anomaly'] = 1

# d. 이벤트 발생 당일(9/7) 및 그 이후 데이터는 학습에서 제외
df_labeled_arm_aze = root_ratio[root_ratio.index < EVENT_DATE].copy()

# =======================================
# 6. 결과 확인 및 저장
# =======================================
print("\n[SUCCESS] 아르메니아-아제르바이잔 (2020-09-07 기준) 레이블링 완료.")
print(f"최종 데이터 Shape: {df_labeled_arm_aze.shape}")

print("\n--- '이상 징후' 시작 지점 데이터 (Label=1) ---")
print(df_labeled_arm_aze[df_labeled_arm_aze['is_anomaly'] == 1].head(3))

print("\n--- '정상' 기간 데이터 (Label=0) ---")
print(df_labeled_arm_aze[df_labeled_arm_aze['is_anomaly'] == 0].head(3))

# 이 데이터프레임을 나중에 다른 사례와 합치기 위해 파일로 저장합니다.
df_labeled_arm_aze.to_csv("labeled_arm_aze.csv")
print("\n[INFO] 'labeled_arm_aze.csv' 파일로 저장 완료.")