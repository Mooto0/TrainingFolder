# =======================================
# 0. 환경 준비
# =======================================
# pip install google-cloud-bigquery pyarrow matplotlib seaborn pandas numpy folium google-auth db-dtypes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import bigquery
from google.oauth2 import service_account
import folium
import matplotlib.dates as mdates # 날짜 형식 지정을 위해 추가

# =======================================
# 1. Google Cloud 인증 (서비스 계정 키 사용)
# =======================================
# !!! 중요: 이 경로는 사용자님의 환경에 맞게 유지해야 합니다 !!!
KEY_PATH = "C:\\AIproject\\my-gdelt-analysis-2025-8990d16f9b7a.json"
credentials = service_account.Credentials.from_service_account_file(KEY_PATH)

# =======================================
# 2. BigQuery에서 데이터 불러오기
# =======================================
# !!! 중요: 이 프로젝트 ID는 사용자님의 ID에 맞게 유지해야 합니다 !!!
PROJECT_ID = "my-gdelt-analysis-2025"
client = bigquery.Client(project=PROJECT_ID, credentials=credentials)

# =======================================
# 3. GDELT 데이터 쿼리 (니제르 2023년)
# =======================================
AGGREGATED_QUERY = """
-- 1단계: 분석에 필요한 최소한의 데이터만 먼저 필터링합니다.
WITH RelevantEvents AS (
  SELECT
    SQLDATE,
    QuadClass,
    EventRootCode
  FROM
    `gdelt-bq.gdeltv2.events`
  WHERE
    -- 기간: 2023년 1월 1일 ~ 2023년 10월 31일
    SQLDATE BETWEEN 20230101 AND 20231031
    -- 국가: 니제르(NER)
    AND (
      Actor1CountryCode IN ('NER') OR
      Actor2CountryCode IN ('NER') OR
      ActionGeo_CountryCode IN ('NER')
    )
)
-- 2단계: 필터링된 데이터를 바탕으로 QuadClass 일일 건수 집계
SELECT
  SQLDATE,
  'QuadClass' as analysis_type,
  CAST(QuadClass AS STRING) as code,
  COUNT(*) as event_count
FROM RelevantEvents
GROUP BY SQLDATE, QuadClass

UNION ALL

-- 3단계: 동일하게 필터링된 데이터를 바탕으로 EventRootCode 일일 건수 집계
SELECT
  SQLDATE,
  'EventRootCode' as analysis_type,
  EventRootCode,
  COUNT(*) as event_count
FROM RelevantEvents
GROUP BY SQLDATE, EventRootCode
"""

print("[INFO] BigQuery에서 집계된 데이터를 불러오는 중입니다 (니제르)...")
df_aggregated = client.query(AGGREGATED_QUERY).to_dataframe()
print("[INFO] 데이터 로딩 완료. Shape:", df_aggregated.shape)

# =======================================
# 4. 데이터 처리 및 비율 계산 (원본과 동일)
# =======================================
df_aggregated['date'] = pd.to_datetime(df_aggregated['SQLDATE'], format='%Y%m%d')

# --- QuadClass 데이터 처리 ---
df_quad = df_aggregated[df_aggregated['analysis_type'] == 'QuadClass']
quad_dist = df_quad.pivot(index='date', columns='code', values='event_count').fillna(0)
quad_dist_ratio = quad_dist.div(quad_dist.sum(axis=1), axis=0).fillna(0)

# --- EventRootCode 데이터 처리 ---
df_root = df_aggregated[df_aggregated['analysis_type'] == 'EventRootCode']
df_root['code'] = df_root['code'].astype(str).str.zfill(2)
root_dist = df_root.pivot(index='date', columns='code', values='event_count').fillna(0)
root_ratio = root_dist.div(root_dist.sum(axis=1), axis=0).fillna(0)

# =======================================
# 5. 시각화 (QuadClass)
# =======================================
fig, ax = plt.subplots(figsize=(14, 6))
# 데이터가 없는 QuadClass가 있을 수 있으므로 방어적으로 1,2,3,4를 순회
for q in ['1', '2', '3', '4']: 
    if q in quad_dist_ratio.columns:
        ax.plot(quad_dist_ratio.index, quad_dist_ratio[q], label=f"QuadClass {int(q)}")

# 2023년 7월 26일에 세로 점선 추가 (니제르 쿠데타)
ax.axvline(pd.to_datetime('2023-07-26'), color='red', linestyle='--', linewidth=1.5, label='Niger Coup (2023-07-26)')

ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.title("Daily QuadClass Ratio (Niger, Jan-Oct 2023)")
plt.xlabel("Date")
plt.ylabel("Ratio")
plt.legend()
plt.tight_layout()
plt.show()

# =======================================
# 6. 전체 EventRootCode 비율 변화 분석
# =======================================
EVENT_ROOT_CODE_LABELS = {
    '01': 'Make Public Statement', '02': 'Appeal', '03': 'Express Intent', '04': 'Consult',
    '05': 'Diplomatic Cooperation', '06': 'Material Cooperation', '07': 'Provide Aid', '08': 'Yield',
    '09': 'Investigate', '10': 'Demand', '11': 'Disapprove', '12': 'Reject',
    '13': 'Threaten', '14': 'Protest', '15': 'Exhibit Force Posture', '16': 'Reduce Relations',
    '17': 'Coerce', '18': 'Assault', '19': 'Fight', '20': 'Use Unconventional Mass Violence'
}

fig, ax = plt.subplots(figsize=(16, 8))
existing_codes = sorted([code for code in EVENT_ROOT_CODE_LABELS.keys() if code in root_ratio.columns])

# Top 5 코드를 강조
top_5_codes = root_ratio.sum().nlargest(5).index

for code in existing_codes:
    if code in top_5_codes:
        ax.plot(root_ratio.index, root_ratio[code], label=f'{code}: {EVENT_ROOT_CODE_LABELS.get(code, "N/A")}', linewidth=2.5)
    else:
        ax.plot(root_ratio.index, root_ratio[code], color='lightgray', linewidth=0.7)

# 2023년 7월 26일에 세로 점선 추가
ax.axvline(pd.to_datetime('2023-07-26'), color='red', linestyle='--', linewidth=1.5, label='Niger Coup (2023-07-26)')

ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.title("Daily Ratio of All EventRootCodes (Niger, Jan-Oct 2023)")
plt.xlabel("Date")
plt.ylabel("Ratio of Total Events")
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.02))
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()