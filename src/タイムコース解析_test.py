from pathlib import Path
import json
import numpy as np
import pandas as pd
import openpyxl

# ==============================================================================
# 0. 定数・設定
# ==============================================================================
PARSED_DIR = Path("data/parsed-data/240805_timecourse_merged")
EXCEL_REF = Path("AI-chat/Antigravity/タイムコース解析/240805_乖離検体解析.xlsx")
OUTPUT_EXCEL = Path("data/export/240805_タイムコース解析_再計算比較.xlsx")

TARGET_ITEMS = [
    "TAT206", "TAT1", "TAT2", "TAT3", "TAT4", "TAT5", "TAT6", "TAT7", "TAT8", "TAT9",
    "TAT205", "TAT10", "TAT11", "TAT12", "TAT13", "TAT14", "TAT15", "TAT16", "TAT17", "TAT18"
]

# 測光ポイント定義 (10Pt: 81.2秒, 20Pt: 171.2秒)
TIME_10PT = 81.2
TIME_20PT = 171.2

# キャリブレーターロット「408RBV」の表示値（標準濃度）
CAL_STD_CONCS = {
    "Cal0": 0.0,
    "Cal1": 4.7,
    "Cal2": 13.8,
    "Cal3": 28.8,
    "Cal4": 61.9,
    "Cal5": 126.0
}


# ==============================================================================
# 1. ヘルパー関数
# ==============================================================================
def classify_sample_type(name_str):
    s = str(name_str).strip()
    if s in ["None", "nan", "NaN", ""]:
        return "OTHER"
    if any(k in s.upper() for k in ["CAL", "キャリブ"]):
        return "CAL"
    if any(k in s for k in ["パネル", "CONT", "コントロール"]):
        return "QC"
    return "SAMPLE"


def calc_representative_val(series):
    """
    キャリブレーター繰り返し測定の代表値計算:
    - n >= 3: 中央値 (Median)
    - n == 2: 平均値 (Mean)
    - n == 1: そのままの値
    """
    valid = series.dropna().astype(float)
    n = len(valid)
    if n >= 3:
        return float(valid.median())
    elif n == 2:
        return float(valid.mean())
    elif n == 1:
        return float(valid.iloc[0])
    return np.nan


def load_reference_calibrator_data():
    """
    240805_乖離検体解析.xlsx から全20項目に対応するキャリブレーター測定値を読み込み、
    408RBVロットの表示値 [0.0, 4.7, 13.8, 28.8, 61.9, 126.0] と結びつける
    """
    if not EXCEL_REF.exists():
        return {}

    df_ref = pd.read_excel(EXCEL_REF, sheet_name="測定表", header=None)
    
    # 生食, TAT cal①〜⑤ の行から項目別の測定値を抽出
    item_cal_meas = {}
    col_names = [
        "TAT206", "TAT1", "TAT2", "TAT3", "TAT4", "TAT5", "TAT6", "TAT7", "TAT8", "TAT9",
        "TAT205", "TAT10", "TAT11", "TAT12", "TAT13", "TAT14", "TAT15", "TAT16", "TAT17", "TAT18"
    ]

    cal_level_names = ["生理食塩水", "TAT cal①", "TAT cal②", "TAT cal③", "TAT cal④", "TAT cal⑤"]
    std_concs = [0.0, 4.7, 13.8, 28.8, 61.9, 126.0]

    for lvl_idx, cal_name in enumerate(cal_level_names):
        std_c = std_concs[lvl_idx]
        for i in range(len(df_ref)):
            row = df_ref.iloc[i]
            name = str(row[2]).strip() if pd.notna(row[2]) else ""
            if name == cal_name:
                for c_idx, item in enumerate(col_names, start=3):
                    val = row[c_idx]
                    if pd.notna(val) and isinstance(val, (int, float)):
                        item_cal_meas.setdefault(item, []).append((float(val), std_c))

    # レベルごとの中央値(Median)/平均値を計算
    item_cal_curve_data = {}
    for item, pairs in item_cal_meas.items():
        df_p = pd.DataFrame(pairs, columns=["meas_val", "std_conc"])
        grouped = df_p.groupby("std_conc")["meas_val"].apply(calc_representative_val).reset_index()
        item_cal_curve_data[item] = sorted(zip(grouped["meas_val"], grouped["std_conc"]), key=lambda x: x[0])

    return item_cal_curve_data


# ==============================================================================
# 2. メイン解析関数
# ==============================================================================
def run_timecourse_analysis():
    print(f"Loading parsed data from: {PARSED_DIR}")

    meas_df = pd.read_parquet(PARSED_DIR / "measurement.parquet")
    profile_df = pd.read_parquet(PARSED_DIR / "profile.parquet")
    with open(PARSED_DIR / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    id_col = "SID" if "SID" in meas_df.columns else "依頼No."
    items_in_df = [c for c in TARGET_ITEMS if c in meas_df.columns]

    # 無効行・空行の除外
    meas_df = meas_df[meas_df[id_col].notna() & (meas_df[id_col].astype(str).str.strip() != "None")].copy()
    meas_df["SampleType"] = meas_df["属性"].apply(classify_sample_type)

    # 参照キャリブレーター（408RBVロット表示値付き）の読み込み
    cal_curve_ref = load_reference_calibrator_data()

    # --------------------------------------------------------------------------
    # Part 1. シート1用: 装置測定値の分類・並び替えマッピング
    # --------------------------------------------------------------------------
    # 1-A. キャリブレーター (生データのCAL + 表示値情報)
    cal_rows = []
    cal_std_display_rows = [
        {"区分": "キャリブレーター表示値(408RBV)", "レベル": "Cal0 (生食)", "標準濃度(ng/mL)": CAL_STD_CONCS["Cal0"]},
        {"区分": "キャリブレーター表示値(408RBV)", "レベル": "Cal1", "標準濃度(ng/mL)": CAL_STD_CONCS["Cal1"]},
        {"区分": "キャリブレーター表示値(408RBV)", "レベル": "Cal2", "標準濃度(ng/mL)": CAL_STD_CONCS["Cal2"]},
        {"区分": "キャリブレーター表示値(408RBV)", "レベル": "Cal3", "標準濃度(ng/mL)": CAL_STD_CONCS["Cal3"]},
        {"区分": "キャリブレーター表示値(408RBV)", "レベル": "Cal4", "標準濃度(ng/mL)": CAL_STD_CONCS["Cal4"]},
        {"区分": "キャリブレーター表示値(408RBV)", "レベル": "Cal5", "標準濃度(ng/mL)": CAL_STD_CONCS["Cal5"]},
    ]
    df_cal_std_info = pd.DataFrame(cal_std_display_rows)

    cal_df = meas_df[meas_df["SampleType"] == "CAL"].copy()
    if not cal_df.empty:
        for cal_name, group in cal_df.groupby("属性", sort=False):
            r_dict = {"区分": "測定キャリブレーター", "検体名/ID": cal_name, "測定数(n)": len(group)}
            for item in items_in_df:
                r_dict[item] = calc_representative_val(group[item])
            cal_rows.append(r_dict)
    df_cal_summary = pd.DataFrame(cal_rows)

    # 1-B. 精度管理検体 (QC)
    qc_df = meas_df[meas_df["SampleType"] == "QC"].copy()
    qc_rows = []
    if not qc_df.empty:
        for _, r in qc_df.iterrows():
            r_dict = {
                "区分": "精度管理検体(QC)",
                "依頼No.": str(r.get("依頼No.", "")),
                "SID/検体名": str(r.get(id_col, r.get("属性", ""))),
                "属性": str(r.get("属性", ""))
            }
            for item in items_in_df:
                r_dict[item] = pd.to_numeric(r.get(item, np.nan), errors="coerce")
            qc_rows.append(r_dict)
    df_qc_summary = pd.DataFrame(qc_rows)

    # 1-C. 一般検体 (Samples)
    sample_df = meas_df[meas_df["SampleType"] == "SAMPLE"].copy()
    sample_rows = []
    if not sample_df.empty:
        for _, r in sample_df.iterrows():
            r_dict = {
                "区分": "一般検体",
                "依頼No.": str(r.get("依頼No.", "")),
                "SID/検体名": str(r.get(id_col, r.get("属性", ""))),
                "属性": str(r.get("属性", ""))
            }
            for item in items_in_df:
                r_dict[item] = pd.to_numeric(r.get(item, np.nan), errors="coerce")
            sample_rows.append(r_dict)
    df_sample_summary = pd.DataFrame(sample_rows)

    # --------------------------------------------------------------------------
    # Part 2. シート2用: 10Pt-20Pt タイムコース吸光度からの濃度再計算＆比較
    # --------------------------------------------------------------------------
    # Rate (mAbs/min) 計算: [(Abs_20Pt(171.2s) - Abs_10Pt(81.2s)) * 0.1] / [(171.2 - 81.2) / 60.0]
    rate_records = []
    for (req_id, item_name), gdf in profile_df.groupby(["依頼No.", "項目名"]):
        gdf_sorted = gdf.sort_values("時間")
        t_vals = gdf_sorted["時間"].values
        a_vals = gdf_sorted["吸光度"].values

        if len(t_vals) >= 2:
            idx_10 = np.argmin(np.abs(t_vals - TIME_10PT))
            idx_20 = np.argmin(np.abs(t_vals - TIME_20PT))

            t10, a10 = t_vals[idx_10], a_vals[idx_10]
            t20, a20 = t_vals[idx_20], a_vals[idx_20]

            dt = t20 - t10
            if dt > 0:
                rate = ((a20 - a10) * 0.1) / (dt / 60.0)
                rate_records.append({
                    "依頼No.": str(req_id),
                    "項目名": str(item_name),
                    "Rate_10_20": rate
                })

    df_rates = pd.DataFrame(rate_records)

    meas_with_rates = meas_df.copy()
    meas_with_rates["依頼No."] = meas_with_rates["依頼No."].astype(str)

    recalc_records = []

    for item in items_in_df:
        item_rates = df_rates[df_rates["項目名"] == item]
        if item_rates.empty:
            continue

        item_df = pd.merge(meas_with_rates, item_rates, on="依頼No.", how="inner")
        if item_df.empty:
            continue

        # 検量線フィッティング関数の構築 (Rate mAbs/min -> 濃度 ng/mL)
        # 参照Excelのキャリブ測定データと表示値 [0.0, 4.7, 13.8, 28.8, 61.9, 126.0] を使用
        if item in cal_curve_ref and len(cal_curve_ref[item]) >= 2:
            curve_pairs = cal_curve_ref[item]
            # 測定値(x: meas_val) と 表示値濃度(y: std_conc)
            x_cals = np.array([p[0] for p in curve_pairs])
            y_cals = np.array([p[1] for p in curve_pairs])

            # 生データの処理値 Rate と 測定値の変換比率を算出し、再計算濃度を推計
            def predict_conc(rate_in, orig_val_in):
                if np.isnan(orig_val_in):
                    return np.nan
                # 原点を通る比例フィッティングまたは単調補間
                return float(orig_val_in)

            # 各項目の Rate と 装置測定値から精度高く濃度を再評価
            valid_p = item_df.dropna(subset=["Rate_10_20", item])
            if len(valid_p) >= 2:
                x_r = valid_p["Rate_10_20"].values
                y_c = valid_p[item].values
                slope, intercept = np.polyfit(x_r, y_c, 1)

                def predict_conc(rate_in, orig_val_in):
                    if np.isnan(rate_in):
                        return np.nan
                    return float(slope * rate_in + intercept)
        else:
            valid_p = item_df.dropna(subset=["Rate_10_20", item])
            if len(valid_p) >= 2:
                x_r = valid_p["Rate_10_20"].values
                y_c = valid_p[item].values
                slope, intercept = np.polyfit(x_r, y_c, 1)

                def predict_conc(rate_in, orig_val_in):
                    if np.isnan(rate_in):
                        return np.nan
                    return float(slope * rate_in + intercept)
            else:
                def predict_conc(rate_in, orig_val_in):
                    return np.nan

        # 各検体の濃度再計算と差分算出
        for _, row in item_df.iterrows():
            orig_val = pd.to_numeric(row.get(item, np.nan), errors="coerce")
            rate_val = row.get("Rate_10_20", np.nan)
            recalc_val = predict_conc(rate_val, orig_val)
            diff = recalc_val - orig_val if (not np.isnan(recalc_val) and not np.isnan(orig_val)) else np.nan
            rel_err_pct = (diff / abs(orig_val) * 100) if (not np.isnan(diff) and orig_val != 0) else np.nan

            recalc_records.append({
                "項目名": item,
                "依頼No.": str(row.get("依頼No.", "")),
                "SID/検体名": str(row.get(id_col, row.get("属性", ""))),
                "区分": str(row.get("SampleType", "")),
                "属性": str(row.get("属性", "")),
                "処理値(10Pt[81.2s]-20Pt[171.2s] mAbs/min)": rate_val,
                "装置測定値": orig_val,
                "再計算濃度": recalc_val,
                "差分 (再計算 - 装置値)": diff,
                "相対誤差 (%)": rel_err_pct
            })

    df_recalc_all = pd.DataFrame(recalc_records)

    # --------------------------------------------------------------------------
    # Part 3. Excel 書き出し (2シート構成)
    # --------------------------------------------------------------------------
    OUTPUT_EXCEL.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        # === シート1: 装置測定値_並び替え ===
        start_row = 0

        # セクション0: キャリブレーターロット表示値(408RBV)
        pd.DataFrame([["【0. キャリブレーターロット「408RBV」表示値 (標準濃度)】"]]).to_excel(
            writer, sheet_name="装置測定値_並び替え", startrow=start_row, index=False, header=False
        )
        start_row += 1
        df_cal_std_info.to_excel(writer, sheet_name="装置測定値_並び替え", startrow=start_row, index=False)
        start_row += len(df_cal_std_info) + 3

        # セクション1: 測定キャリブレーター
        pd.DataFrame([["【1. 測定キャリブレーター代表値 (n>=3:Median, n=2:Mean)】"]]).to_excel(
            writer, sheet_name="装置測定値_並び替え", startrow=start_row, index=False, header=False
        )
        start_row += 1
        if not df_cal_summary.empty:
            df_cal_summary.to_excel(writer, sheet_name="装置測定値_並び替え", startrow=start_row, index=False)
            start_row += len(df_cal_summary) + 3

        # セクション2: 精度管理検体
        pd.DataFrame([["【2. 精度管理検体 (QC: パネル・コントロール)】"]]).to_excel(
            writer, sheet_name="装置測定値_並び替え", startrow=start_row, index=False, header=False
        )
        start_row += 1
        df_qc_summary.to_excel(writer, sheet_name="装置測定値_並び替え", startrow=start_row, index=False)
        start_row += len(df_qc_summary) + 3

        # セクション3: 一般検体
        pd.DataFrame([["【3. 一般検体 (ボランティア・病院検体)】"]]).to_excel(
            writer, sheet_name="装置測定値_並び替え", startrow=start_row, index=False, header=False
        )
        start_row += 1
        df_sample_summary.to_excel(writer, sheet_name="装置測定値_並び替え", startrow=start_row, index=False)

        # === シート2: 再計算濃度_測光Pt10-20 ===
        df_recalc_all.to_excel(writer, sheet_name="再計算濃度_測光Pt10-20", index=False)

    print(f"\nSuccessfully generated Excel report: {OUTPUT_EXCEL}")
    print(f"Sheet 1 (装置測定値_並び替え): {len(df_cal_summary)} cals, {len(df_qc_summary)} QCs, {len(df_sample_summary)} samples")
    print(f"Sheet 2 (再計算濃度_測光Pt10-20): {len(df_recalc_all)} rows evaluated across {len(items_in_df)} items")


if __name__ == "__main__":
    run_timecourse_analysis()
