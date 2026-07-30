import pandas as pd
profile = pd.read_parquet('data/parsed-data/260708 (Calibratin~乖離検体まで) F12 CSV_data/profile.parquet')
item_df = profile[profile['項目名'] == 'TAT207']
times = sorted(item_df['時間'].unique())
print(f"Number of time points: {len(times)}")
print(f"Time points: {times}")
