import pandas as pd
profile = pd.read_parquet('data/parsed-data/260708 (Calibratin~乖離検体まで) F12 CSV_data/profile.parquet')
item = "TAT207"
df_item = profile[profile['項目名']==item]
for req in ['C001', 'C002', 'C003', 'C145', 'C146', 'C147', 'C148', 'C149', 'C150']:
    print(f"Req: {req}, len: {len(df_item[df_item['依頼No.']==req])}, first val: {df_item[df_item['依頼No.']==req]['処理値'].iloc[0] if len(df_item[df_item['依頼No.']==req]) else 'NA'}")
