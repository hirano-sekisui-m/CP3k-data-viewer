import pandas as pd
meas = pd.read_parquet('data/parsed-data/260708 (Calibratin~乖離検体まで) F12 CSV_data/measurement.parquet')
print(meas[['依頼No.', 'TAT207']].head(20))
