import pandas as pd
import numpy as np
import glob
import os
import sys

# =======================================
# 1. 파일 검색
# =======================================
file_pattern = "labeled_*.csv"
csv_files = glob.glob(file_pattern)

if not csv_files:
    print(f"[ERROR] '{file_pattern}' 패턴과 일치하는 CSV 파일을 찾을 수 없습니다.")
    sys.exit()

print(f"[INFO] 총 {len(csv_files)}개의 'labeled_*.csv' 파일을 찾았습니다:")
for f in sorted(csv_files):
    print(f"       - {f}")

# =======================================
# 2. 모든 CSV 파일 불러오기 및 병합
# =======================================
all_dataframes = []
for file in csv_files:
    try:
        df = pd.read_csv(file, index_col='date', parse_dates=True)
        
        # [🚨 핵심 수정 🚨]
        # 파일명에서 'case' 이름을 추출하여 새 컬럼으로 추가합니다.
        # 예: "labeled_israel_palestine.csv" -> "israel_palestine"
        case_name = file.replace("labeled_", "").replace(".csv", "")
        df['case'] = case_name
        # [🚨 수정 끝 🚨]
        
        all_dataframes.append(df)
    except Exception as e:
        print(f"[WARNING] '{file}' 파일을 불러오는 중 오류 발생: {e}")

if not all_dataframes:
    print("[ERROR] 불러올 수 있는 데이터프레임이 없습니다.")
    sys.exit()

print("\n[INFO] 모든 데이터프레임을 하나로 병합합니다...")
final_dataset = pd.concat(all_dataframes)

# =======================================
# 3. 정렬 및 최종 확인
# =======================================
# 'case'별로 먼저 정렬한 뒤, 'date'별로 정렬합니다.
# (concat 후에 sort_index()만 해도 날짜가 섞이진 않지만, case별로 정렬하는 것이 더 명확합니다.)
final_dataset = final_dataset.sort_values(by=['case', 'date'])

print(f"\n[INFO] 모든 데이터 병합 완료!")
print(f"       최종 데이터셋 Shape: {final_dataset.shape}")
print(f"       'case' 컬럼 고유 값: {final_dataset['case'].nunique()}개")

# =======================================
# 4. 통합 데이터셋 파일로 저장
# =======================================
output_filename = "unified_dataset_with_case.csv" # <--- 파일 이름 변경
print(f"\n[INFO] 통합된 데이터셋을 '{output_filename}'로 저장합니다...")
final_dataset.to_csv(output_filename, index=True) 

print(f"[SUCCESS] 저장 완료! '{output_filename}'가 생성되었습니다.")