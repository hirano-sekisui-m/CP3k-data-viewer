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
OUTPUT_EXCEL = Path("data/export/240805_タイムコース解析_測光ポイント変動再計算.xlsx")

TARGET_ITEMS = [
    "TAT206", "TAT1", "TAT2", "TAT3", "TAT4", "TAT5", "TAT6", "TAT7", "TAT8", "TAT9",
    "TAT205", "TAT10", "TAT11", "TAT12", "TAT13", "TAT14", "TAT15", "TAT16", "TAT17", "TAT18"
]

# 評価する測光ポイントパターン (名称, 開始Pt, 終了Pt)
PT_PATTERNS = [
    ("2-12", 2, 12),
    ("4-14", 4, 14),
    ("6-16", 6, 16),
    ("8-18", 8, 18),
    ("10-20", 10, 20),
    ("装置生データ", None, None),
    ("2-21", 2, 21),
    ("4-21", 4, 21),
    ("6-21", 6, 21),
]

# 各ポイントの定義秒数
PT_TIMES = {
    1: 0.0, 2: 9.2, 3: 18.0, 4: 27.2, 5: 36.0, 6: 45.2, 7: 54.0, 8: 63.2, 9: 72.0, 10: 81.2,
    11: 90.0, 12: 99.2, 13: 108.0, 14: 117.2, 15: 126.0, 16: 135.2, 17: 144.0, 18: 153.2,
    19: 162.0, 20: 171.2, 21: 180.0
}

# キャリブレーターロット「408RBV」の表示値濃度 (Cal0 〜 Cal5)
STD_CONCS = [0.0, 4.7, 13.8, 28.8, 61.9, 126.0]


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


def calc_rate_mabs_min(time_vals, abs_vals, pt_start, pt_end):
    """
    指定ポイント (pt_start ~ pt_end) における吸光度変化率 mAbs/min を計算
    """
    if pt_start not in PT_TIMES or pt_end not in PT_TIMES:
        return np.nan

    target_t1 = PT_TIMES[pt_start]
    target_t2 = PT_TIMES[pt_end]

    idx_1 = np.argmin(np.abs(time_vals - target_t1))
    idx_2 = np.argmin(np.abs(time_vals - target_t2))

    t1, a1 = time_vals[idx_1], abs_vals[idx_1]
    t2, a2 = time_vals[idx_2], abs_vals[idx_2]

    dt = t2 - t1
    if dt <= 0:
        return np.nan

    # Rate (mAbs/min)
    return float(((a2 - a1) * 0.1) / (dt / 60.0))


def load_calibrator_raw_curves_multi_pt(profile_df, clean_cal):
    """
    全測光Ptパターンについて、全20項目のキャリブ点のRate (mAbs/min) を計算し、
    表示値 [0.0, 4.7, 13.8, 28.8, 61.9, 126.0] との区間折れ線辞書を作成
    """
    cal_curves_by_pat = {}

    # clean_cal (キャリブ生データ) の C001~C357 行
    # 依頼No. と 項目名 で各パターンの Rate を算出
    for pat_name, pt_s, pt_e in PT_PATTERNS:
        if pat_name == "装置生データ":
            continue

        item_curves = {}
        for item_name, group in clean_cal.groupby("項目名"):
            valid_group = group.dropna(subset=["処理値"]).copy()
            valid_group["c_num"] = valid_group["依頼No."].str.extract(r"C?(\d+)")[0].astype(float)
            valid_group = valid_group.sort_values("c_num")

            # 原則 18行 (6レベル x 3回測定)
            # 各行のプロファイル吸光度時系列を取得
            rates = []
            for _, r_row in valid_group.iterrows():
                # 依頼No.に対応するプロファイルデータをprofile_dfから検索、またはシートの吸光度列を使用
                req_no = str(r_row["依頼No."])
                match_prof = profile_df[(profile_df["依頼No."].astype(str) == req_no) & (profile_df["項目名"].astype(str) == item_name)]
                
                if not match_prof.empty:
                    m_sorted = match_prof.sort_values("時間")
                    rate_v = calc_rate_mabs_min(m_sorted["時間"].values, m_sorted["吸光度"].values, pt_s, pt_e)
                else:
                    # デフォルト処理値 (10-20) の比率等
                    rate_v = pd.to_numeric(r_row["処理値"], errors="coerce")

                if not np.isnan(rate_v):
                    rates.append(rate_v)

            if len(rates) >= 6:
                n_levels = min(6, len(rates) // 3)
                cal_rates = []
                for l in range(n_levels):
                    sub_rates = rates[l * 3 : (l + 1) * 3]
                    cal_rates.append(float(np.median(sub_rates)))

                if len(cal_rates) == 6:
                    pairs = sorted(zip(cal_rates, STD_CONCS), key=lambda x: x[0])
                    r_arr = np.array([p[0] for p in pairs])
                    c_arr = np.array([p[1] for p in pairs])
                    item_curves[item_name] = (r_arr, c_arr)

        cal_curves_by_pat[pat_name] = item_curves

    return cal_curves_by_pat


# ==============================================================================
# 2. メイン解析・横長データ構築関数
# ==============================================================================
def run_multi_pt_analysis():
    print(f"Loading parsed data from: {PARSED_DIR}")

    meas_df = pd.read_parquet(PARSED_DIR / "measurement.parquet")
    profile_df = pd.read_parquet(PARSED_DIR / "profile.parquet")
    with open(PARSED_DIR / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # 参照Excel「キャリブ生データ」の読み込み
    df_cal = pd.read_excel(EXCEL_REF, sheet_name="キャリブ生データ")
    clean_cal = df_cal.dropna(subset=["依頼No.", "項目名"]).copy()
    clean_cal["依頼No."] = clean_cal["依頼No."].astype(str).str.strip()
    clean_cal["項目名"] = clean_cal["項目名"].astype(str).str.strip()
    clean_cal["処理値"] = pd.to_numeric(clean_cal["処理値"], errors="coerce")

    id_col = "SID" if "SID" in meas_df.columns else "依頼No."
    items_in_df = [c for c in TARGET_ITEMS if c in meas_df.columns]

    # 無効行の除外
    meas_df = meas_df[meas_df[id_col].notna() & (meas_df[id_col].astype(str).str.strip() != "None")].copy()
    meas_df["SampleType"] = meas_df["属性"].apply(classify_sample_type)
    meas_df["依頼No."] = meas_df["依頼No."].astype(str)

    print("Building calibration curves for all 9 photometric patterns...")
    cal_curves_by_pat = load_calibrator_raw_curves_multi_pt(profile_df, clean_cal)

    # 各パターン別・検体別・項目別の Rate を事前に計算
    print("Calculating Rate (mAbs/min) for all samples across all patterns...")
    rates_by_pat = {}
    for pat_name, pt_s, pt_e in PT_PATTERNS:
        if pat_name == "装置生データ":
            continue

        records = []
        for (req_id, item_name), gdf in profile_df.groupby(["依頼No.", "項目名"]):
            gdf_sorted = gdf.sort_values("時間")
            rate = calc_rate_mabs_min(gdf_sorted["時間"].values, gdf_sorted["吸光度"].values, pt_s, pt_e)
            records.append({"依頼No.": str(req_id), "項目名": str(item_name), "Rate": rate})

        rates_by_pat[pat_name] = pd.DataFrame(records)

    # --------------------------------------------------------------------------
    # 横長ワイドフォーマット DataFrame の構築
    # --------------------------------------------------------------------------
    print("Constructing Wide-Format matrix (1 row per sample, 180+ columns)...")
    
    # 1検体1行の基本情報
    samples_base = meas_df[["依頼No.", id_col, "SampleType", "属性"]].copy()
    samples_base.columns = ["依頼No.", "SID/検体名", "区分", "属性"]
    samples_base = samples_base.drop_duplicates(subset=["依頼No."]).reset_index(drop=True)

    # 横長マトリクス用の辞書データを作成
    wide_data = []

    for _, s_row in samples_base.iterrows():
        req_id = s_row["依頼No."]
        row_dict = {
            "依頼No.": req_id,
            "SID/検体名": s_row["SID/検体名"],
            "区分": s_row["区分"],
            "属性": s_row["属性"]
        }

        # 元データの該当行
        m_row = meas_df[meas_df["依頼No."] == req_id]
        if m_row.empty:
            continue
        m_row = m_row.iloc[0]

        # 20項目 x 9パターンの濃度再計算値を並べる
        for item in items_in_df:
            orig_val = pd.to_numeric(m_row.get(item, np.nan), errors="coerce")

            for pat_name, pt_s, pt_e in PT_PATTERNS:
                col_key = f"{item}_{pat_name}"

                if pat_name == "装置生データ":
                    row_dict[col_key] = orig_val
                else:
                    # 当該パターンの Rate を取得
                    df_p_rates = rates_by_pat[pat_name]
                    match_r = df_p_rates[(df_p_rates["依頼No."] == req_id) & (df_p_rates["項目名"] == item)]
                    
                    if not match_r.empty:
                        rate_val = match_r.iloc[0]["Rate"]
                    else:
                        rate_val = np.nan

                    # 検量線 (Piecewise Linear) 補間
                    curves = cal_curves_by_pat.get(pat_name, {})
                    if item in curves and not np.isnan(rate_val):
                        r_cals, c_cals = curves[item]
                        recalc_val = float(np.interp(rate_val, r_cals, c_cals))
                    else:
                        recalc_val = np.nan

                    row_dict[col_key] = recalc_val

        wide_data.append(row_dict)

    df_wide = pd.DataFrame(wide_data)

    # --------------------------------------------------------------------------
    # シート2用: キャリブRate一覧 (パターン別) の構築
    # --------------------------------------------------------------------------
    cal_rate_summary_rows = []
    for pat_name, curves in cal_curves_by_pat.items():
        for item, (r_cals, c_cals) in curves.items():
            r_dict = {"測光Ptパターン": pat_name, "項目名": item}
            for idx, (r_v, c_v) in enumerate(zip(r_cals, c_cals)):
                r_dict[f"Cal{idx}_表示値({c_v}ng/mL)"] = r_v
            cal_rate_summary_rows.append(r_dict)

    df_cal_rates_summary = pd.DataFrame(cal_rate_summary_rows)

    # --------------------------------------------------------------------------
    # Excel ファイル出力 (2シート)
    # --------------------------------------------------------------------------
    OUTPUT_EXCEL.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        # === シート1: 測光Pt変動_濃度再計算マトリクス ===
        df_wide.to_excel(writer, sheet_name="測光Pt変動_濃度再計算マトリクス", index=False)

        # === シート2: キャリブRate_Pt別 ===
        df_cal_rates_summary.to_excel(writer, sheet_name="キャリブRate_Pt別", index=False)

    print(f"\nSuccessfully generated Wide-Format Excel report: {OUTPUT_EXCEL}")
    print(f"Sheet 1 (測光Pt変動_濃度再計算マトリクス): {df_wide.shape[0]} rows x {df_wide.shape[1]} columns")
    print(f"Sheet 2 (キャリブRate_Pt別): {len(df_cal_rates_summary)} rows")


if __name__ == "__main__":
    run_multi_pt_analysis()
