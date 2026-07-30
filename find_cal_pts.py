import pandas as pd
profile = pd.read_parquet('data/parsed-data/260708 (Calibratin~乖離検体まで) F12 CSV_data/profile.parquet')

# User said: defaults 10 to 20 for C145 ~ C150
# Wait, for TAT207 C001, C002, C003 the processing values are 2.3, 9.3, 14.3.
# Are they 10 to 20 too?
def calc_proc(req, item, p_start=10, p_end=20):
    df = profile[(profile['項目名']==item) & (profile['依頼No.']==req)].sort_values('時間')
    if df.empty: return None
    t10 = df.iloc[p_start-1]
    t20 = df.iloc[p_end-1]
    proc = ((t20['吸光度'] - t10['吸光度']) * 0.1) / ((t20['時間'] - t10['時間']) / 60.0)
    return proc

print("C001 default calc:", calc_proc('C001', 'TAT207', 10, 20))
print("C002 default calc:", calc_proc('C002', 'TAT207', 10, 20))
print("C003 default calc:", calc_proc('C003', 'TAT207', 10, 20))
