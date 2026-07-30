import pandas as pd
import numpy as np

profile = pd.read_parquet('data/parsed-data/260708 (Calibratin~乖離検体まで) F12 CSV_data/profile.parquet')

def calc_proc(req, item, p_start=10, p_end=20):
    df = profile[(profile['項目名']==item) & (profile['依頼No.']==req)].sort_values('時間')
    if df.empty: return None
    t10 = df.iloc[p_start-1]
    t20 = df.iloc[p_end-1]
    proc = ((t20['吸光度'] - t10['吸光度']) * 0.1) / ((t20['時間'] - t10['時間']) / 60.0)
    return proc

print(calc_proc('0002', 'TAT207'))
print(calc_proc('0003', 'TAT207'))
print(calc_proc('0004', 'TAT207'))
