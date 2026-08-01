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

# イレギュラー補完用 Request-0001 (依頼No. 1, 2, 3)
CAL5_EXTRA_REQUEST_IDS = ["0001", "0002", "0003", "1", "2", "3"]


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

    return float(((a2 - a1) * 0.1) / (dt / 60.0))


def calc_representative_val_flexible(arr):
    """
    測定数 n に合わせた代表値算出:
    - n >= 3: 中央値 (Median)
    - n == 2: 平均値 (Mean)
    - n == 1: 生値そのまま
    """
    n = len(arr)
    if n >= 3:
        return float(np.median(arr[:3]))
    elif n == 2:
        return float(np.mean(arr))
    elif n == 1:
        return float(arr[0])
    return np.nan


def load_calibrator_raw_curves_multi_pt(profile_df, clean_cal, meas_df):
    """
    「キャリブ生データ」シートの ^C\\d+$ ID に加え、
    8/1測定の 'TAT CAL5' (Request-0001: 依頼No. 1, 2, 3) データを Cal5 (126.0 ng/mL) として補完。
    全9測光Ptパターンについて Cal0〜Cal5 の区間折れ線補間辞書を作成。
    """
    c_only_cal = clean_cal[clean_cal["依頼No."].str.match(r"^C\d+$")].copy()
    c_only_cal["c_num"] = c_only_cal["依頼No."].str.extract(r"C(\d+)")[0].astype(int)

    cal_curves_by_pat = {}

    for pat_name, pt_s, pt_e in PT_PATTERNS:
        if pat_name == "装置生データ":
            continue

        item_curves = {}
        for item_name in TARGET_ITEMS:
            sub = c_only_cal[c_only_cal["項目名"] == item_name].sort_values("c_num").dropna(subset=["処理値"])
            rates = []

            if not sub.empty:
                for _, r_row in sub.iterrows():
                    req_no = str(r_row["依頼No."])
                    match_prof = profile_df[(profile_df["依頼No."].astype(str) == req_no) & (profile_df["項目名"].astype(str) == item_name)]

                    if not match_prof.empty:
                        m_sorted = match_prof.sort_values("時間")
                        rate_v = calc_rate_mabs_min(m_sorted["時間"].values, m_sorted["吸光度"].values, pt_s, pt_e)
                    else:
                        rate_v = pd.to_numeric(r_row["処理値"], errors="coerce")

                    if not np.isnan(rate_v):
                        rates.append(rate_v)

            # 各3行刻みでレベル代表値を集約 (Cal0 ~ Cal4/Cal5)
            cal_rates = []
            idx = 0
            while idx < len(rates) and len(cal_rates) < 6:
                rem = len(rates) - idx
                if rem <= 2 and len(cal_rates) == 5:
                    chunk = rates[idx:]
                else:
                    chunk = rates[idx : idx + 3]

                val = calc_representative_val_flexible(chunk)
                if not np.isnan(val):
                    cal_rates.append(val)
                idx += len(chunk)

            # もし Cal5 (6番目のレベル) が不足している場合、Request-0001 (TAT CAL5: 依頼No. 0001, 0002, 0003) から穴埋め
            if len(cal_rates) < 6:
                cal5_prof = profile_df[
                    (profile_df["依頼No."].astype(str).str.zfill(4).isin(["0001", "0002", "0003"])) &
                    (profile_df["項目名"].astype(str) == item_name)
                ]
                if not cal5_prof.empty:
                    c5_rates = []
                    for req_id_c5, c5_gdf in cal5_prof.groupby("依頼No."):
                        c5_sorted = c5_gdf.sort_values("時間")
                        rv = calc_rate_mabs_min(c5_sorted["時間"].values, c5_sorted["吸光度"].values, pt_s, pt_e)
                        if not np.isnan(rv):
                            c5_rates.append(rv)

                    if c5_rates:
                        cal5_val = calc_representative_val_flexible(c5_rates)
                        if not np.isnan(cal5_val):
                            # Cal5 (126.0 ng/mL) として追加
                            if len(cal_rates) == 5:
                                cal_rates.append(cal5_val)
                            elif len(cal_rates) < 5:
                                while len(cal_rates) < 5:
                                    cal_rates.append(np.nan)
                                cal_rates.append(cal5_val)

            # 有効なレベルと表示値濃度の割り当て
            c_concs = STD_CONCS[:len(cal_rates)]

            valid_pairs = [(r, c) for r, c in zip(cal_rates, c_concs) if not np.isnan(r) and not np.isnan(c)]
            if len(valid_pairs) >= 2:
                pairs = sorted(valid_pairs, key=lambda x: x[0])
                r_arr = np.array([p[0] for p in pairs])
                c_arr = np.array([p[1] for p in pairs])
                item_curves[item_name] = (r_arr, c_arr)

        cal_curves_by_pat[pat_name] = item_curves

    return cal_curves_by_pat


def interpolate_or_extrapolate(r_in, r_cals, c_cals):
    """
    区間折れ線補間 (Piecewise Linear) + 外挿処理
    """
    if np.isnan(r_in):
        return np.nan

    if r_cals[0] <= r_in <= r_cals[-1]:
        return float(np.interp(r_in, r_cals, c_cals))

    if r_in < r_cals[0]:
        if len(r_cals) >= 2 and (r_cals[1] - r_cals[0]) != 0:
            slope = (c_cals[1] - c_cals[0]) / (r_cals[1] - r_cals[0])
            return float(c_cals[0] + slope * (r_in - r_cals[0]))
        return float(c_cals[0])

    if r_in > r_cals[-1]:
        if len(r_cals) >= 2 and (r_cals[-1] - r_cals[-2]) != 0:
            slope = (c_cals[-1] - c_cals[-2]) / (r_cals[-1] - r_cals[-2])
            return float(c_cals[-1] + slope * (r_in - r_cals[-1]))
        return float(c_cals[-1])

    return float(np.interp(r_in, r_cals, c_cals))


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

    print("Building calibrator curves with Cal5 extra补完 (Request-0001) for all 9 patterns across ALL 20 items...")
    cal_curves_by_pat = load_calibrator_raw_curves_multi_pt(profile_df, clean_cal, meas_df)

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

    samples_base = meas_df[["依頼No.", id_col, "SampleType", "属性"]].copy()
    samples_base.columns = ["依頼No.", "SID/検体名", "区分", "属性"]
    samples_base = samples_base.drop_duplicates(subset=["依頼No."]).reset_index(drop=True)

    wide_data = []

    for _, s_row in samples_base.iterrows():
        req_id = s_row["依頼No."]
        row_dict = {
            "依頼No.": req_id,
            "SID/検体名": s_row["SID/検体名"],
            "区分": s_row["区分"],
            "属性": s_row["属性"]
        }

        m_row = meas_df[meas_df["依頼No."] == req_id]
        if m_row.empty:
            continue
        m_row = m_row.iloc[0]

        for item in items_in_df:
            orig_val = pd.to_numeric(m_row.get(item, np.nan), errors="coerce")

            for pat_name, pt_s, pt_e in PT_PATTERNS:
                col_key = f"{item}_{pat_name}"

                if pat_name == "装置生データ":
                    row_dict[col_key] = orig_val
                else:
                    df_p_rates = rates_by_pat[pat_name]
                    match_r = df_p_rates[(df_p_rates["依頼No."] == req_id) & (df_p_rates["項目名"] == item)]

                    if not match_r.empty:
                        rate_val = match_r.iloc[0]["Rate"]
                    else:
                        rate_val = np.nan

                    curves = cal_curves_by_pat.get(pat_name, {})
                    if item in curves and not np.isnan(rate_val):
                        r_cals, c_cals = curves[item]
                        recalc_val = interpolate_or_extrapolate(rate_val, r_cals, c_cals)
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
        df_wide.to_excel(writer, sheet_name="測光Pt変動_濃度再計算マトリクス", index=False)
        df_cal_rates_summary.to_excel(writer, sheet_name="キャリブRate_Pt別", index=False)

    print(f"\nSuccessfully generated Wide-Format Excel report: {OUTPUT_EXCEL}")
    print(f"Sheet 1 (測光Pt変動_濃度再計算マトリクス): {df_wide.shape[0]} rows x {df_wide.shape[1]} columns")
    print(f"Sheet 2 (キャリブRate_Pt別): {len(df_cal_rates_summary)} rows")


if __name__ == "__main__":
    run_multi_pt_analysis()
