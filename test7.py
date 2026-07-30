import pandas as pd
profile = pd.read_parquet('data/parsed-data/260708 (Calibratin~乖離検体まで) F12 CSV_data/profile.parquet')
print(profile[profile['依頼No.'].isin(['C145', 'C146', 'C147', 'C148', 'C149', 'C150']) & (profile['項目名']=='TAT207')].head(20))
