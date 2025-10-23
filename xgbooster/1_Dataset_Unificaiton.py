import pandas as pd
import numpy as np
import glob
import os
import sys

# =======================================
# 1. 파일 검색
# =======================================
# 스크립트가 실행되는 현재 폴더에서 'labeled_*.csv' 패턴의 모든 파일을 찾습니다.
file_pattern = "labeled_*.csv"
csv_files = glob.glob(file_pattern)

if not csv_files:
    print(f"[ERROR] '{file_pattern}' 패턴과 일치하는 CSV 파일을 찾을 수 없습니다.")
    print("       이 스크립트가 'labeled_*.csv' 파일들과 동일한 폴더에 있는지 확인하세요.")
    sys.exit() # 파일이 없으면 스크립트 종료

print(f"[INFO] 총 {len(csv_files)}개의 'labeled_*.csv' 파일을 찾았습니다:")
for f in sorted(csv_files):
    print(f"       - {f}")

# =======================================
# 2. 모든 CSV 파일 불러오기 및 병합
# =======================================
all_dataframes = []
for file in csv_files:
    try:
        # 'date' 컬럼을 인덱스로 사용하고, 날짜 형식으로 불러옵니다.
        df = pd.read_csv(file, index_col='date', parse_dates=True)
        all_dataframes.append(df)
    except Exception as e:
        print(f"[WARNING] '{file}' 파일을 불러오는 중 오류 발생: {e}")

if not all_dataframes:
    print("[ERROR] 불러올 수 있는 데이터프레임이 없습니다.")
    sys.exit()

print("\n[INFO] 모든 데이터프레임을 하나로 병합합니다...")
# pd.concat을 사용하여 모든 데이터프레임을 위아래로 합칩니다.
final_dataset = pd.concat(all_dataframes)

# =======================================
# 3. 정렬 및 최종 확인
# =======================================
# 날짜(인덱스) 기준으로 오름차순 정렬합니다.
final_dataset = final_dataset.sort_index()

print(f"\n[INFO] 모든 데이터 병합 완료!")
print(f"       최종 데이터셋 Shape: {final_dataset.shape}")
print(f"       데이터 기간: {final_dataset.index.min().date()} ~ {final_dataset.index.max().date()}")

print("\n[INFO] 최종 데이터셋 'is_anomaly' 분포:")
print(final_dataset['is_anomaly'].value_counts())

# =======================================
# 4. [요청 사항] 통합 데이터셋 파일로 저장
# =======================================
output_filename = "unified_dataset.csv"
print(f"\n[INFO] 통합된 데이터셋을 '{output_filename}'로 저장합니다...")

# index=True를 설정해야 'date' 인덱스가 파일에 함께 저장됩니다.
final_dataset.to_csv(output_filename, index=True) 

print(f"[SUCCESS] 저장 완료! '{output_filename}'가 생성되었습니다.")
print("       이제 2단계(피처 엔지니어링)를 진행할 수 있습니다.")