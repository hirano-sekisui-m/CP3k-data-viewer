import pandas as pd
meas = pd.read_parquet('data/parsed-data/260708 (Calibratin~乖離検体まで) F12 CSV_data/measurement.parquet')
item = "TAT207"
cals = meas[meas['依頼No.'].str.startswith('C')].dropna(subset=[item])[['依頼No.', item]]
print(cals.head(20))
