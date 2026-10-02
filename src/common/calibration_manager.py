import numpy as np
import pandas as pd
import json
import re
from pathlib import Path
from scipy.interpolate import interp1d

# === Calibrator Detection ===

def is_calibrator(request_no, attribute=None, mode="id_pattern", keywords=None) -> bool:
    """
    Check if a given request number belongs to a calibrator.
    """
    if keywords is None:
        keywords = ["CAL", "cal", "キャリブ"]
        
    is_id_match = False
    if mode in ("id_pattern", "both"):
        is_id_match = bool(re.match(r"^C\d+$", str(request_no), re.IGNORECASE))
        
    is_attr_match = False
    if mode in ("attribute", "both") and attribute is not None:
        is_attr_match = any(kw in str(attribute) for kw in keywords)
        
    if mode == "id_pattern":
        return is_id_match
    elif mode == "attribute":
        return is_attr_match
    elif mode == "both":
        return is_id_match or is_attr_match
    return False

def detect_calibrators(measurement_df, profile_df, item_name, mode="id_pattern", keywords=None) -> list[str]:
    """
    Detect all calibrator IDs for a given item.
    """
    req_nos = set()
    
    # Extract IDs from profile_df
    if profile_df is not None and not profile_df.empty:
        if '項目名' in profile_df.columns and '依頼No.' in profile_df.columns:
            subset = profile_df[profile_df['項目名'] == item_name]
            req_nos.update(subset['依頼No.'].dropna().astype(str).tolist())
            
    # Extract IDs from measurement_df
    attr_col_name = None
    if measurement_df is not None and not measurement_df.empty:
        if '属性' in measurement_df.columns:
            attr_col_name = '属性'
        elif len(measurement_df.columns) >= 5:
            attr_col_name = measurement_df.columns[4]
            
        found_item_col = None
        for col in measurement_df.columns:
            if item_name in col and 'FLAG' not in col:
                found_item_col = col
                break
                
        if found_item_col:
            subset = measurement_df[measurement_df[found_item_col].notna()]
            req_nos.update(subset['依頼No.'].dropna().astype(str).tolist())
            
    # Filter using is_calibrator
    cal_ids = []
    for req in req_nos:
        attr_val = None
        if measurement_df is not None and not measurement_df.empty and attr_col_name is not None:
            row = measurement_df[measurement_df['依頼No.'].astype(str) == str(req)]
            if not row.empty:
                attr_val = row.iloc[0][attr_col_name]
                
        if is_calibrator(req, attribute=attr_val, mode=mode, keywords=keywords):
            cal_ids.append(req)
            
    return sorted(list(cal_ids))

# === Calibration Table Construction ===

def build_cal_level_table(cal_ids, n_levels, n_replicates):
    """
    Build a 2D table of calibrator IDs.
    """
    table = []
    idx = 0
    warning = None
    
    if len(cal_ids) != n_levels * n_replicates:
        warning = f"Expected {n_levels * n_replicates} IDs, but got {len(cal_ids)}."
        
    for i in range(n_levels):
        row = []
        for j in range(n_replicates):
            if idx < len(cal_ids):
                row.append(str(cal_ids[idx]))
            else:
                row.append("")
            idx += 1
        table.append(row)
        
    return table, warning

# === Rate Calculations ===

def calc_rate(profile_df, request_no, item_name, time_start, time_end) -> float:
    """
    Calculate the rate (mAbs/min) for a single sample.
    """
    subset = profile_df[(profile_df['依頼No.'].astype(str) == str(request_no)) & 
                        (profile_df['項目名'] == item_name)]
    if subset.empty:
        return np.nan
        
    times = subset['時間'].values
    abss = subset['吸光度'].values
    
    if len(times) == 0:
        return np.nan
        
    idx_start = np.argmin(np.abs(times - time_start))
    idx_end = np.argmin(np.abs(times - time_end))
    
    t_start_actual = times[idx_start]
    t_end_actual = times[idx_end]
    a_start = abss[idx_start]
    a_end = abss[idx_end]
    
    if t_end_actual == t_start_actual:
        return np.nan
        
    rate = ((a_end - a_start) * 0.1) / ((t_end_actual - t_start_actual) / 60.0)
    return rate

def calc_rates_batch(profile_df, item_name, time_start, time_end) -> dict:
    """
    全サンプルについて指定項目の処理値(Rate)を一括算出する。
    """
    subset = profile_df[profile_df['項目名'] == item_name]
    req_nos = subset['依頼No.'].dropna().unique()
    
    rates = {}
    for req in req_nos:
        rate = calc_rate(profile_df, req, item_name, time_start, time_end)
        rates[str(req)] = rate
    return rates

def aggregate_cal_rates(rates_dict, level_table, method="median") -> list[float]:
    """
    Compute representative rate per calibration level.
    """
    agg_rates = []
    for row in level_table:
        vals = []
        for req in row:
            if req and req in rates_dict and not np.isnan(rates_dict[req]):
                vals.append(rates_dict[req])
                
        if len(vals) == 0:
            agg_rates.append(np.nan)
        elif len(vals) == 1:
            agg_rates.append(vals[0])
        elif len(vals) == 2:
            agg_rates.append(float(np.mean(vals)))
        else:
            if method == "mean":
                agg_rates.append(float(np.mean(vals)))
            else:
                agg_rates.append(float(np.median(vals)))
    return agg_rates

# === Curve Construction and Prediction ===

def build_calibration_curve(cal_rates, concentrations, curve_mode="piecewise_linear") -> dict:
    """
    Build a calibration curve dictionary.
    """
    valid_rates = []
    valid_concs = []
    for r, c in zip(cal_rates, concentrations):
        if not np.isnan(r) and not np.isnan(c):
            valid_rates.append(r)
            valid_concs.append(c)
            
    if len(valid_rates) < 2:
        return None
        
    rates = np.array(valid_rates)
    concs = np.array(valid_concs)
    
    sort_idx = np.argsort(rates)
    rates = rates[sort_idx]
    concs = concs[sort_idx]
    
    interp_func = None
    if curve_mode == "spline":
        if len(rates) >= 4:
            interp_func = interp1d(rates, concs, kind='cubic', fill_value='extrapolate')
        else:
            curve_mode = "piecewise_linear"
            
    return {
        "rates": rates,
        "concentrations": concs,
        "curve_mode": curve_mode,
        "interp_func": interp_func
    }

def predict_concentration(cal_curve, rate_value) -> float:
    """
    Predict concentration from a rate value using the calibration curve.
    """
    if cal_curve is None or np.isnan(rate_value):
        return np.nan
        
    if cal_curve["curve_mode"] == "spline" and cal_curve["interp_func"] is not None:
        return float(cal_curve["interp_func"](rate_value))
    else: # piecewise_linear
        rates = cal_curve["rates"]
        concs = cal_curve["concentrations"]
        
        if rate_value < rates[0]:
            slope = (concs[1] - concs[0]) / (rates[1] - rates[0]) if rates[1] != rates[0] else 0
            return float(concs[0] + slope * (rate_value - rates[0]))
        elif rate_value > rates[-1]:
            slope = (concs[-1] - concs[-2]) / (rates[-1] - rates[-2]) if rates[-1] != rates[-2] else 0
            return float(concs[-1] + slope * (rate_value - rates[-1]))
        else:
            return float(np.interp(rate_value, rates, concs))

def recalculate_all_samples(profile_df, measurement_df, item_name, cal_curve, time_start, time_end, cal_id_to_conc=None) -> pd.DataFrame:
    """
    Recalculate concentrations for all samples.
    """
    rates_dict = calc_rates_batch(profile_df, item_name, time_start, time_end)
    
    attr_col_name = None
    if measurement_df is not None and not measurement_df.empty:
        if '属性' in measurement_df.columns:
            attr_col_name = '属性'
        elif len(measurement_df.columns) >= 5:
            attr_col_name = measurement_df.columns[4]
            
    meas_col = None
    if measurement_df is not None and not measurement_df.empty:
        for col in measurement_df.columns:
            if item_name in col and 'FLAG' not in col:
                meas_col = col
                break
                
    results = []
    
    for req_no, rate_val in rates_dict.items():
        attr_val = None
        orig_conc = np.nan
        
        if measurement_df is not None and not measurement_df.empty:
            row = measurement_df[measurement_df['依頼No.'].astype(str) == req_no]
            if not row.empty:
                if attr_col_name:
                    attr_val = row.iloc[0][attr_col_name]
                if meas_col:
                    orig_conc = row.iloc[0][meas_col]
                    
        is_cal = is_calibrator(req_no, attribute=attr_val, mode="both")
        sample_type = "キャリブレーター" if is_cal else "一般検体"
        
        if is_cal and cal_id_to_conc and req_no in cal_id_to_conc:
            orig_conc = cal_id_to_conc[req_no]
            
        recalc_conc = predict_concentration(cal_curve, rate_val)
        
        results.append({
            "依頼No.": req_no,
            "サンプル区分": sample_type,
            "処理値": rate_val,
            "装置測定値": orig_conc,
            "再計算濃度": recalc_conc
        })
        
    return pd.DataFrame(results)

# === Configuration & UI Helpers ===

def save_cal_config(config, path):
    """Save calibration config to JSON."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def load_cal_config(path) -> dict:
    """Load calibration config from JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_cal_level_detail_table(rates_dict, level_table, concentrations, item_name) -> pd.DataFrame:
    """
    Build a detail DataFrame showing each individual calibrator measurement.
    """
    records = []
    for level_idx, row in enumerate(level_table):
        conc = concentrations[level_idx] if level_idx < len(concentrations) else np.nan
        for req in row:
            if req:
                rate = rates_dict.get(req, np.nan)
                records.append({
                    "レベル": f"Cal {level_idx}",
                    "依頼No.": req,
                    "表示値濃度": conc,
                    "処理値(Rate)": rate
                })
    return pd.DataFrame(records)


def get_cal_level_summary_table(rates_dict, level_table, concentrations, agg_method="median") -> pd.DataFrame:
    """
    各レベルの代表値、n数、Mean、SD、CV%(n>=2)をまとめたサマリーテーブルを作成。
    キャリブレーションの安定性・バラつき（CV%）確認に使用。
    """
    rows = []
    for level_idx, row in enumerate(level_table):
        conc = concentrations[level_idx] if level_idx < len(concentrations) else np.nan
        vals = [rates_dict[req] for req in row if req and req in rates_dict and np.isfinite(rates_dict[req])]
        n = len(vals)
        mean_val = float(np.mean(vals)) if n > 0 else np.nan
        sd_val = float(np.std(vals, ddof=1)) if n >= 2 else np.nan
        cv_val = float((sd_val / mean_val) * 100.0) if (np.isfinite(sd_val) and mean_val != 0) else np.nan
        rep_val = float(np.median(vals)) if agg_method == "median" and n > 0 else mean_val

        rows.append({
            "レベル": f"Cal {level_idx}",
            "表示値濃度": conc,
            "測定点数(n)": n,
            "代表値(Rate)": rep_val,
            "平均値(Mean)": mean_val,
            "標準偏差(SD)": sd_val,
            "CV(%)": cv_val
        })
    return pd.DataFrame(rows)


def compare_two_recalc_results(df_a, df_b, label_a="Lot-A", label_b="Lot-B") -> pd.DataFrame:
    """
    2つの再計算結果（ロット間または条件間）をマージして差や比率を算出。
    実検体・コントロール検体におけるロット間測定値差の検討に使用。
    """
    if df_a is None or df_b is None or df_a.empty or df_b.empty:
        return pd.DataFrame()

    sub_a = df_a[["依頼No.", "サンプル区分", "再計算濃度"]].rename(columns={"再計算濃度": f"濃度_{label_a}"})
    sub_b = df_b[["依頼No.", "再計算濃度"]].rename(columns={"再計算濃度": f"濃度_{label_b}"})

    merged = sub_a.merge(sub_b, on="依頼No.", how="inner")
    ca = pd.to_numeric(merged[f"濃度_{label_a}"], errors="coerce")
    cb = pd.to_numeric(merged[f"濃度_{label_b}"], errors="coerce")

    merged[f"差({label_b}-{label_a})"] = cb - ca
    merged[f"相対比({label_b}/{label_a})"] = np.where(ca != 0, cb / ca, np.nan)
    merged[f"乖離率(%)"] = np.where(ca != 0, (cb - ca) / np.abs(ca) * 100.0, np.nan)

    return merged
