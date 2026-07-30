import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

meas = pd.read_parquet('data/parsed-data/260708 (Calibratin~乖離検体まで) F12 CSV_data/measurement.parquet')
profile = pd.read_parquet('data/parsed-data/260708 (Calibratin~乖離検体まで) F12 CSV_data/profile.parquet')

# For TAT207, calibrator requests C145 to C150 seem to correspond to Cal1 to Cal6
orig_proc_vals = [2.6, 9.7, 14.3, 19.9, 29.3, 51.1]
cal_concs = [0.0, 5.3, 14.0, 30.4, 56.7, 139.8]

# From problem description:
# 検量線モード：折れ線(他には、スプラインや直線などがあります)

# If it's a polyline (piecewise linear interpolation), we interpolate conc based on proc val
def interpolate_conc(y, x_points, y_points):
    return np.interp(y, y_points, x_points)

print("Check piecewise interpolation against original general sample concs:")
gen_reqs = ['0002', '0003', '0004']
for req in gen_reqs:
    y = profile[(profile['項目名']=='TAT207') & (profile['依頼No.']==req)]['処理値'].iloc[0]
    orig_conc = meas[meas['依頼No.']==req]['TAT207'].iloc[0]
    calc_conc = interpolate_conc(y, cal_concs, orig_proc_vals)
    print(f"Req {req}: proc {y}, orig conc {orig_conc}, calc conc {calc_conc}")
