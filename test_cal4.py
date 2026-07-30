import pandas as pd
import numpy as np

profile = pd.read_parquet('data/parsed-data/260708 (Calibratin~乖離検体まで) F12 CSV_data/profile.parquet')
df_item = profile[profile["項目名"] == "TAT207"]

def get_proc_value(req_id, p_start, p_end):
    df_req = df_item[df_item["依頼No."] == req_id].sort_values("時間")
    if len(df_req) < max(p_start, p_end):
        return np.nan
    row_start = df_req.iloc[p_start - 1]
    row_end = df_req.iloc[p_end - 1]
    return ((row_end["吸光度"] - row_start["吸光度"]) * 0.1) / ((row_end["時間"] - row_start["時間"]) / 60.0)

print("C145 proc:", get_proc_value("C145", 10, 20))
print("C146 proc:", get_proc_value("C146", 10, 20))
print("0001 proc:", get_proc_value("0001", 10, 20))
