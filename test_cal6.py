import pandas as pd
import numpy as np

# We'll need a function to compute processing value based on start and end point.
# "測光ポイント(デフォルト)：10から20ポイント目" -> indices 9 and 19
# Then build piecewise linear/spline calibration curves from C001, C002... Cxxx

profile = pd.read_parquet('data/parsed-data/260708 (Calibratin~乖離検体まで) F12 CSV_data/profile.parquet')

item = "TAT207"
df_item = profile[profile['項目名']==item]

cals = ["C145", "C146", "C147", "C148", "C149", "C150"]
t10 = df_item[df_item['時間']==81.2].set_index('依頼No.')['吸光度']
t20 = df_item[df_item['時間']==171.2].set_index('依頼No.')['吸光度']

proc = ((t20 - t10) * 0.1) / ((171.2 - 81.2) / 60.0)
print(proc.loc[cals])

t5 = df_item[df_item['時間']==36.0].set_index('依頼No.')['吸光度']
t15 = df_item[df_item['時間']==126.0].set_index('依頼No.')['吸光度']
proc_new = ((t15 - t5) * 0.1) / ((126.0 - 36.0) / 60.0)
print("New proc values for 5 to 15:")
print(proc_new.loc[cals])
