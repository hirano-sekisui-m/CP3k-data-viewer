import pandas as pd
profile = pd.read_parquet('data/parsed-data/260708 (Calibratin~乖離検体まで) F12 CSV_data/profile.parquet')
print(profile[(profile['項目名']=='TAT207') & (profile['依頼No.'].str.startswith('C'))]['依頼No.'].unique())
