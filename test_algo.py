import pandas as pd
import numpy as np

profile = pd.read_parquet('data/parsed-data/260708 (Calibratin~乖離検体まで) F12 CSV_data/profile.parquet')
meas = pd.read_parquet('data/parsed-data/260708 (Calibratin~乖離検体まで) F12 CSV_data/measurement.parquet')

item = "TAT207"
profile_item = profile[profile['項目名'] == item]

cal_requests = ["C145", "C146", "C147", "C148", "C149", "C150"]
# The ones we saw have non-null TAT207 in meas:
# 37  C145     2.6
# 38  C146     9.7
# 39  C147    14.3
# 40  C148    19.9
# 41  C149    29.3
# 42  C150    51.1

# Are these the calibrators? Let's check their profile
print("Original concs for C145-C150:")
print(meas[meas['依頼No.'].isin(cal_requests)][['依頼No.', item]])

cal_concs = [0.0, 5.3, 14.0, 30.4, 56.7, 139.8] # Given as example

p_start = 10
p_end = 20

times = sorted(profile_item["時間"].unique())
t_start = times[p_start - 1]
t_end = times[p_end - 1]
print(f"Time points: {t_start}s to {t_end}s")

df_start = profile_item[profile_item["時間"] == t_start].set_index("依頼No.")["吸光度"]
df_end = profile_item[profile_item["時間"] == t_end].set_index("依頼No.")["吸光度"]

time_diff_min = (t_end - t_start) / 60.0
proc_vals = ((df_end - df_start) * 0.1) / time_diff_min

print("Computed processing values for calibrators:")
for req in cal_requests:
    print(req, proc_vals.get(req, np.nan))

orig_proc = profile_item.groupby("依頼No.")["処理値"].first()
print("Original processing values for calibrators:")
for req in cal_requests:
    print(req, orig_proc.get(req, np.nan))
