import pandas as pd
meas = pd.read_parquet('data/parsed-data/260708 (Calibratin~乖離検体まで) F12 CSV_data/measurement.parquet')
print(meas[meas['依頼No.'].isin(['C145', 'C146', 'C147', 'C148', 'C149', 'C150'])][['依頼No.', 'TAT207', 'TAT207_FLAG']])
