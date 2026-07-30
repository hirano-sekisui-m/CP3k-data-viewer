import sys
from pathlib import Path
import json
import traceback
import itertools
import math
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common.csv_loader import load_parsed_for_analysis
from common.analysis_utils import (
    setup_japanese_font, pick_col, normalize_group_col, detect_value_cols,
    safe_name, make_output_dirs, apply_value_range_filter,
    pearson_r, regression_fit_info, plot_suite, write_df_to_sheet,
    compute_pair_sample_metrics, classify_outlier_level, insert_images_into_excel,
    REGRESSION_METHODS_ALL, SHEET_PLOTS, SHEET_SUMMARY,
    SHEET_OUTLIERS, SHEET_SAMPLE_METRICS,
    OUT_SUFFIX, ID_COL_CANDIDATES, GROUP_COL_CANDIDATES, VALUE_PREFIXES
)

setup_japanese_font()

OUTPUT_ROOT = PROJECT_ROOT / "data" / "export"

st.set_page_config(page_title="相関解析 & タイムコース表示", layout="wide")

# ============================================================
# Session State Initialization
# ============================================================
if "df" not in st.session_state:
    st.session_state.update({
        "df": None,
        "id_col": None,
        "group_col": None,
        "value_cols": None,
        "parsed_dir": None,
        "metadata": None,
        "profile_df": None,
        "ref_outlier_map": None,
        "analysis_results": None,
        "metadata_enhanced": None
    })

# ============================================================
# Calibrator Detection
# ============================================================
def is_calibrator(request_no: str) -> bool:
    """依頼No.が 'C + 数字' パターンならキャリブレーターと判定する。
    例: 'C001', 'C1', 'c002' -> True
        '0001', '001'       -> False
    """
    return bool(re.match(r'^C\d+$', str(request_no).strip(), re.IGNORECASE))


def find_calibrator_groups(measurement_df, item_col):
    """指定項目に対して、連続するキャリブレーターIDグループを検出する。"""
    if measurement_df is None or item_col not in measurement_df.columns:
        return []
    
    groups = []
    current_group = []
    current_values = []
    
    id_col = "依頼No." if "依頼No." in measurement_df.columns else measurement_df.columns[0]
    
    for _, row in measurement_df.iterrows():
        rid = str(row[id_col])
        is_cal = is_calibrator(rid)
        val = row.get(item_col)
        has_val = pd.notna(val)
        
        if is_cal and has_val:
            current_group.append(rid)
            current_values.append(float(val))
        else:
            if current_group:
                groups.append({'ids': current_group, 'values': current_values})
                current_group, current_values = [], []
    if current_group:
        groups.append({'ids': current_group, 'values': current_values})
    return groups


# ============================================================
# Helper Functions
# ============================================================
def discover_latest_parsed_dir(parsed_root=None):
    root = PROJECT_ROOT
    parsed_root = Path(parsed_root) if parsed_root is not None else root / "data" / "parsed-data"
    if not parsed_root.is_absolute():
        parsed_root = root / parsed_root

    if parsed_root.is_dir() and (parsed_root / "measurement.parquet").exists() and (parsed_root / "metadata.json").exists():
        return [parsed_root]

    if not parsed_root.exists():
        return None

    candidates = [
        p for p in parsed_root.iterdir()
        if p.is_dir() and (p / "measurement.parquet").exists() and (p / "metadata.json").exists()
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates

def load_parsed_data_for_app(parsed_dir):
    parsed_dir = Path(parsed_dir)
    measurement_df, profile_df, metadata, prescription_columns = load_parsed_for_analysis(parsed_dir)

    measurement_df = measurement_df.copy()

    # SampleID列が存在しても中身が全てNone/nanの場合はSIDや依頼No.で上書きする
    def _sid_col_is_empty(df):
        if "SampleID" not in df.columns:
            return True
        s = df["SampleID"].astype(str).str.strip()
        return s.isin(["None", "nan", "NaN", "", "<NA>"]).all()

    if _sid_col_is_empty(measurement_df):
        if "SID" in measurement_df.columns:
            measurement_df["SampleID"] = measurement_df["SID"].astype(str)
        elif "依頼No." in measurement_df.columns:
            measurement_df["SampleID"] = measurement_df["依頼No."].astype(str)
        else:
            measurement_df["SampleID"] = measurement_df.index.astype(str)

    # SIDもID候補に追加して確実に検出（ID_COL_CANDIDATESにSIDがない場合への対策）
    id_col_candidates_ext = list(ID_COL_CANDIDATES) + ["SID", "依頼No."]
    id_col = pick_col(measurement_df, id_col_candidates_ext, default="SampleID")
    group_col_raw = pick_col(measurement_df, GROUP_COL_CANDIDATES, default=None)
    group_col = normalize_group_col(measurement_df, group_col_raw)

    value_cols = [c for c in prescription_columns if c in measurement_df.columns]
    if not value_cols:
        value_cols = detect_value_cols(measurement_df, id_col, group_col, prefixes=VALUE_PREFIXES)
    value_cols = [c for c in value_cols if c in measurement_df.columns and not str(c).endswith("_FLAG")]

    return measurement_df, profile_df, metadata, parsed_dir, id_col, group_col, value_cols

def load_action(target_dir):
    try:
        df, profile_df, metadata, parsed_dir, id_col, group_col, value_cols = load_parsed_data_for_app(target_dir)
        st.session_state.update({
            "df": df,
            "profile_df": profile_df,
            "metadata": metadata,
            "parsed_dir": parsed_dir,
            "id_col": id_col,
            "group_col": group_col,
            "value_cols": value_cols,
            "analysis_results": None,
            "metadata_enhanced": None,
            "ref_outlier_map": None
        })
        st.success(f"読み込み完了: {parsed_dir.name}\nID列: {id_col}, 比較列数: {len(value_cols)}")
    except Exception as e:
        st.error(f"データ読み込みエラー:\n{traceback.format_exc()}")

# ============================================================
# Main UI
# ============================================================
st.title("相関解析 & タイムコース表示")
st.markdown("""
<div style='border:1px solid #ccc; padding:12px; border-radius:8px; background:#fafafa; line-height:1.6;'>
<b>このツールの目的</b><br>
解析済みデータ（parsed-data）から相関・回帰・Bland–Altman・残差・乖離候補を確認し、Excelに出力します。<br>
また、検体ごとのタイムコース反応（吸光度変化）を確認できます。
</div>
""", unsafe_allow_html=True)

# 1. Directory Selection & Loading
st.header("① データ読み込み")
parsed_dirs = discover_latest_parsed_dir()
if parsed_dirs:
    dir_options = {p.name: str(p) for p in parsed_dirs}
    selected_dir_name = st.selectbox("解析対象", options=list(dir_options.keys()))
    selected_dir_path = dir_options[selected_dir_name]
    if st.button("①データ読み込み", type="primary"):
        with st.spinner("読み込み中..."):
            load_action(selected_dir_path)
else:
    st.warning("対象データが見つかりません。")

st.divider()

if st.session_state["df"] is not None:
    value_cols = st.session_state["value_cols"]

    tab1, tab2, tab3, tab4 = st.tabs(["相関解析", "Excel出力", "タイムコース表示", "タイムコース解析"])

    # ----------------------------------------------------
    # TAB 1: 相関解析
    # ----------------------------------------------------
    with tab1:
        st.header("② 解析設定 & 実行")
        col1, col2, col3 = st.columns(3)
        with col1:
            mode = st.selectbox("モード", options=["all", "adjacent", "baseline"],
                                format_func=lambda x: {"all":"全組合せ", "adjacent":"隣同士のみ", "baseline":"基準処方 vs その他"}[x], index=2)
            if mode == "baseline":
                baseline_sel = st.multiselect("基準処方", options=value_cols, default=[value_cols[0]] if value_cols else [])
            else:
                baseline_sel = []
        with col2:
            reg_method = st.selectbox("回帰法", options=["OLS", "Deming", "TheilSen", "PassingBablok"],
                                      format_func=lambda x: {"OLS":"OLS（最小二乗）", "Deming":"Deming（両軸誤差）", "TheilSen":"Theil-Sen（ロバスト）", "PassingBablok":"Passing-Bablok（ノンパラメトリック）"}[x], index=3)
            all_reg_ck = st.checkbox("全回帰法で出力", value=False)
            deming_lambda_val = st.number_input("λ(Deming)", value=1.0)
        with col3:
            outlier_mode_dd = st.selectbox("乖離判定基準", options=["zMAD", "error"], format_func=lambda x: "zMAD (標準化残差)" if x == "zMAD" else "error (臨床的許容誤差)")
            if outlier_mode_dd == "zMAD":
                z_thresh_val = st.slider("乖離z(MAD)", min_value=1.5, max_value=8.0, value=3.5, step=0.1)
                pct_thresh_val, abs_thresh_val = 10.0, 2.0
            else:
                pct_thresh_val = st.number_input("許容誤差(%)", value=10.0)
                abs_thresh_val = st.number_input("許容誤差(絶対値)", value=2.0)
                z_thresh_val = 3.5
            label_top_val = st.slider("ラベル数", min_value=0, max_value=30, value=8, step=1)

        st.markdown("**範囲絞り設定**")
        col4, col5, col6 = st.columns(3)
        with col4:
            use_range_ck = st.checkbox("対象範囲絞り", value=False)
        with col5:
            range_min_txt = st.number_input("下限", value=0.0)
        with col6:
            range_max_txt = st.number_input("上限", value=100.0)

        st.markdown("**乖離判定の基準（このペアで赤い検体を他でも赤く表示）**")
        col7, col8 = st.columns(2)
        with col7:
            ref_pair_x = st.selectbox("乖離基準X", options=["(未選択)"] + value_cols, index=1 if len(value_cols) >= 2 else 0)
        with col8:
            ref_pair_y = st.selectbox("乖離基準Y", options=["(未選択)"] + value_cols, index=2 if len(value_cols) >= 2 else 0)

        col9, col10 = st.columns(2)
        with col9:
            show_py_ck = st.checkbox("画面上に表示", value=True)
        with col10:
            max_show_val = st.slider("表示上限", min_value=0, max_value=30, value=30, step=1)

        if st.button("②解析実行", type="primary", key="run_analysis"):
            with st.spinner("解析実行中..."):
                df = st.session_state["df"]
                id_col = st.session_state["id_col"]
                group_col = st.session_state["group_col"]
                parsed_dir = st.session_state["parsed_dir"]

                if mode == "adjacent":
                    pairs = list(zip(value_cols[:-1], value_cols[1:]))
                elif mode == "all":
                    pairs = list(itertools.combinations(value_cols, 2))
                else:
                    bases = list(baseline_sel) if baseline_sel else [value_cols[0]]
                    pairs = []
                    for b in bases:
                        for c in value_cols:
                            if c != b and (b, c) not in pairs and (c, b) not in pairs:
                                pairs.append((b, c))

                methods = REGRESSION_METHODS_ALL if all_reg_ck else [reg_method]
                lam = deming_lambda_val
                z_thresh = z_thresh_val

                summary_rows, outlier_tables, sample_metric_tables, figures = [], [], [], []
                shown = 0
                run_metadata = st.session_state["metadata"].copy() if st.session_state["metadata"] else {}

                # --- Pre-calculate Reference Outliers for Coloring ---
                ref_outlier_map = {}
                ref_x, ref_y = ref_pair_x, ref_pair_y
                if ref_x != "(未選択)" and ref_y != "(未選択)" and ref_x != ref_y:
                    df_ref = apply_value_range_filter(df, ref_x, ref_y, use_range=use_range_ck, lo=range_min_txt, hi=range_max_txt)
                    if id_col in df_ref.columns:
                        df_ref[id_col] = df_ref[id_col].astype(str).replace(["nan", "None", "<NA>", "NaN"], "Unknown")
                        df_ref[id_col] = df_ref[id_col].fillna("Unknown")
                    sub_ref = df_ref[[ref_x, ref_y]].dropna()
                    if len(sub_ref) >= 2:
                        xr, yr = sub_ref[ref_x].astype(float).values, sub_ref[ref_y].astype(float).values
                        ar, br, _ = regression_fit_info(xr, yr, method=reg_method, deming_lambda=lam)
                        try:
                            metrics_ref = compute_pair_sample_metrics(df_ref, id_col, group_col, ref_x, ref_y, ar, br, z_thresh=z_thresh_val, outlier_mode=outlier_mode_dd, pct_thresh=pct_thresh_val, abs_thresh=abs_thresh_val)
                        except TypeError:
                            metrics_ref = compute_pair_sample_metrics(df_ref, id_col, group_col, ref_x, ref_y, ar, br, z_thresh=z_thresh_val)
                        for _, row in metrics_ref.iterrows():
                            ref_outlier_map[str(row[id_col])] = row["outlier_level"]

                for method in methods:
                    for xcol, ycol in pairs:
                        try:
                            df_pair = apply_value_range_filter(df, xcol, ycol, use_range=use_range_ck, lo=range_min_txt, hi=range_max_txt)
                            if id_col in df_pair.columns:
                                df_pair[id_col] = df_pair[id_col].astype(str).replace(["nan", "None", "<NA>", "NaN"], "Unknown")
                                df_pair[id_col] = df_pair[id_col].fillna("Unknown")
                            sub = df_pair[[xcol, ycol]].dropna()
                            if len(sub) < 2:
                                st.warning(f"警告: {xcol} vs {ycol} の有効データが2件未満のためスキップします。")
                                continue

                            x, y = sub[xcol].astype(float).values, sub[ycol].astype(float).values
                            r = pearson_r(x, y)
                            a, b, fit_info = regression_fit_info(x, y, method=method, deming_lambda=lam)

                            pair_key = f"{xcol}_vs_{ycol}"
                            try:
                                metrics_df = compute_pair_sample_metrics(df_pair, id_col, group_col, xcol, ycol, a, b, z_thresh=z_thresh_val, outlier_mode=outlier_mode_dd, pct_thresh=pct_thresh_val, abs_thresh=abs_thresh_val)
                            except TypeError:
                                metrics_df = compute_pair_sample_metrics(df_pair, id_col, group_col, xcol, ycol, a, b, z_thresh=z_thresh_val)

                            color_list = []
                            for _, row in metrics_df.iterrows():
                                sid = str(row[id_col])
                                sample_meta = run_metadata.setdefault(sid, {})
                                outliers_meta = sample_meta.setdefault("outliers", {})

                                if outlier_mode_dd == "error":
                                    level = row.get("outlier_level", "none")
                                    outliers_meta[pair_key] = {"level": level, "abs_residual": float(row.get("abs_residual", np.nan)), "rel_diff_yx_pct": float(row.get("rel_diff_yx_pct", np.nan))}
                                else:
                                    z_mad = row.get("z_MAD", np.nan)
                                    level = classify_outlier_level(abs(z_mad), thresh=z_thresh_val) if np.isfinite(z_mad) else "none"
                                    outliers_meta[pair_key] = {"level": level, "z_MAD": float(z_mad) if np.isfinite(z_mad) else None}

                                target_level = ref_outlier_map.get(sid, "none") if ref_outlier_map else level

                                if target_level == "strong_candidate": color_list.append("red")
                                elif target_level == "candidate": color_list.append("orange")
                                elif target_level == "mild_candidate": color_list.append("yellow")
                                else: color_list.append("#1f77b4")

                            ref_keys = [k for k, v in ref_outlier_map.items() if v in ("strong_candidate", "candidate", "mild_candidate")] if ref_outlier_map else None
                            try:
                                fig, used_sub, flagged, bias, loa = plot_suite(
                                    df=df_pair, id_col=id_col, group_col=group_col, xcol=xcol, ycol=ycol,
                                    method=method, lam=lam, a=a, b=b, r=r, fit_info=fit_info,
                                    z_thresh=z_thresh_val, outlier_label_top=label_top_val,
                                    fig_width=16, fig_height=10, dpi=100, external_colors=color_list,
                                    force_flagged_ids=ref_keys, outlier_mode=outlier_mode_dd, pct_thresh=pct_thresh_val, abs_thresh=abs_thresh_val
                                )
                            except TypeError:
                                fig, used_sub, flagged, bias, loa = plot_suite(
                                    df=df_pair, id_col=id_col, group_col=group_col, xcol=xcol, ycol=ycol,
                                    method=method, lam=lam, a=a, b=b, r=r, fit_info=fit_info,
                                    z_thresh=z_thresh_val, outlier_label_top=label_top_val,
                                    fig_width=16, fig_height=10, dpi=100, external_colors=color_list,
                                    force_flagged_ids=ref_keys
                                )

                            if fig is not None:
                                figures.append((fig, method, xcol, ycol))
                            else:
                                # Fallback generation if plot_suite unexpectedly returned None
                                import matplotlib.pyplot as plt
                                fallback_fig, fallback_ax = plt.subplots(figsize=(8, 6))
                                fallback_ax.scatter(x, y, alpha=0.7)
                                fallback_ax.set_title(f"Fallback Plot: {xcol} vs {ycol}")
                                fallback_ax.set_xlabel(xcol)
                                fallback_ax.set_ylabel(ycol)
                                figures.append((fallback_fig, method, xcol, ycol))
                                if bias is None: bias = float(np.nanmean(y - x))
                                if flagged is None: flagged = pd.DataFrame()
                                st.warning(f"警告: {xcol} vs {ycol} の描画処理で予期せぬ空データが返されたため、フォールバック描画を行いました。")

                            summary_rows.append({"regression": method, "X": xcol, "Y": ycol, "n": len(x), "r": r,
                                                 "slope": a, "intercept": b, "BA_bias": bias if bias is not None and not np.isnan(bias) else None, "n_outliers": len(flagged) if flagged is not None else 0})
                            if not metrics_df.empty: sample_metric_tables.append(metrics_df.assign(regression=method, X=xcol, Y=ycol))
                            if flagged is not None and not flagged.empty: outlier_tables.append(flagged.assign(regression=method, X=xcol, Y=ycol))
                        except Exception as e:
                            st.error(f"{xcol} vs {ycol} の解析中にエラーが発生しました: {e}")

                # 解析設定をまとめて保存（Excel出力時の情報表示用）
                run_settings = {
                    "parsed_dir": str(parsed_dir),
                    "input_stem": parsed_dir.name,
                    "mode": mode,
                    "regression_method": reg_method,
                    "all_regression": all_reg_ck,
                    "deming_lambda": deming_lambda_val,
                    "outlier_mode": outlier_mode_dd,
                    "z_thresh": z_thresh_val,
                    "pct_thresh": pct_thresh_val,
                    "abs_thresh": abs_thresh_val,
                    "outlier_label_top": label_top_val,
                    "use_range": use_range_ck,
                    "range_min": range_min_txt,
                    "range_max": range_max_txt,
                    "ref_pair_x": ref_pair_x,
                    "ref_pair_y": ref_pair_y,
                    "n_pairs": len(pairs),
                    "pairs": [(x, y) for x, y in pairs],
                }

                st.session_state["analysis_results"] = {
                    "run_metadata": run_metadata,
                    "ref_outlier_map": ref_outlier_map,
                    "summary_rows": summary_rows,
                    "sample_metric_tables": sample_metric_tables,
                    "outlier_tables": outlier_tables,
                    "figures": figures,
                    "run_settings": run_settings,
                }
                st.session_state["metadata_enhanced"] = run_metadata
                st.session_state["ref_outlier_map"] = ref_outlier_map

                st.success(f"解析完了! {len(summary_rows)}件のペアを処理しました。内容を確認後、必要であれば『Excel出力』タブへ進んでください。")

        if st.session_state.get("analysis_results") and show_py_ck:
            st.markdown("### 解析結果のグラフ")
            shown = 0
            for fig, method, xcol, ycol in st.session_state["analysis_results"]["figures"]:
                if shown < max_show_val:
                    st.pyplot(fig)
                    shown += 1
                else:
                    break

    # ----------------------------------------------------
    # TAB 2: Excel出力
    # ----------------------------------------------------
    with tab2:
        st.header("③ Excel出力")
        if st.session_state["analysis_results"] is None:
            st.warning("先に『相関解析』タブで『②解析実行』を行ってください。")
        else:
            # 解析設定の概要を常に表示
            res = st.session_state["analysis_results"]
            s = res.get("run_settings", {})
            if s:
                _mode_labels = {"adjacent": "隣接ペア", "all": "全ペア", "baseline": "基準ペア"}
                _outlier_labels = {"zMAD": "zMAD (標準化残差)", "error": "error (臨床的許容誤差)"}
                _reg_labels = {"OLS": "OLS（最小二乗）", "Deming": "Deming（両軸誤差）",
                               "TheilSen": "Theil-Sen（ロバスト）", "PassingBablok": "Passing-Bablok（ノンパラメトリック）"}
                _ref_x = s.get("ref_pair_x", "(未選択)")
                _ref_y = s.get("ref_pair_y", "(未選択)")
                _ref_str = f"{_ref_x} vs {_ref_y}" if _ref_x != "(未選択)" and _ref_y != "(未選択)" else "なし"

                if s.get("outlier_mode") == "zMAD":
                    _outlier_detail = f"z(MAD) ≥ {s.get('z_thresh', 3.5)}"
                else:
                    _outlier_detail = f"許容誤差 {s.get('pct_thresh', 10.0)}% / 絶対値 {s.get('abs_thresh', 2.0)}"

                st.info(
                    f"📋 **解析設定の確認**\n\n"
                    f"| 項目 | 値 |\n"
                    f"|---|---|\n"
                    f"| 📂 解析対象ファイル | `{s.get('input_stem', '?')}` |\n"
                    f"| 🔗 ペアモード | {_mode_labels.get(s.get('mode',''), s.get('mode','?'))} ({s.get('n_pairs', '?')} ペア) |\n"
                    f"| 📐 回帰法 | {_reg_labels.get(s.get('regression_method',''), s.get('regression_method','?'))}{' (全回帰法)' if s.get('all_regression') else ''} |\n"
                    f"| 🔍 乖離判定基準 | {_outlier_labels.get(s.get('outlier_mode',''), '?')} ({_outlier_detail}) |\n"
                    f"| 🎯 乖離基準ペア | {_ref_str} |\n"
                    f"| 📏 対象範囲絞り | {'あり: ' + str(s.get('range_min','')) + ' ～ ' + str(s.get('range_max','')) if s.get('use_range') else 'なし'} |"
                )

            if st.button("③Excel出力", type="primary"):
                with st.spinner("Excelファイル作成中..."):
                    try:
                        parsed_dir = st.session_state["parsed_dir"]

                        dirs = make_output_dirs(OUTPUT_ROOT, input_stem=parsed_dir.name)
                        img_paths = []

                        for fig, method, xcol, ycol in res["figures"]:
                            png = dirs["plots"] / f"QC_{safe_name(method)}_{safe_name(ycol)}_vs_{safe_name(xcol)}.png"
                            fig.savefig(png, bbox_inches="tight")
                            img_paths.append(png)

                        with open(parsed_dir / "metadata.json", "w", encoding="utf-8") as f:
                            json.dump(res["run_metadata"], f, indent=2, ensure_ascii=False)

                        output_xlsx = dirs["excel"] / f"{parsed_dir.name}{OUT_SUFFIX}.xlsx"
                        import openpyxl
                        wb = openpyxl.Workbook()
                        wb.save(output_xlsx)

                        if img_paths:
                            try:
                                insert_images_into_excel(input_xlsx=output_xlsx, output_xlsx=output_xlsx, image_paths=img_paths, plot_sheet=SHEET_PLOTS)
                            except Exception as e:
                                st.warning(f"Plots could not be inserted into Excel: {e}")

                        if res["summary_rows"]: write_df_to_sheet(output_xlsx, pd.DataFrame(res["summary_rows"]), SHEET_SUMMARY)
                        if res["sample_metric_tables"]: write_df_to_sheet(output_xlsx, pd.concat(res["sample_metric_tables"]), SHEET_SAMPLE_METRICS)
                        if res["outlier_tables"]: write_df_to_sheet(output_xlsx, pd.concat(res["outlier_tables"]), SHEET_OUTLIERS)

                        # 出力完了メッセージ（解析設定 + 出力先をまとめて表示）
                        st.success(f"✅ Excel保存完了!")
                        st.info(
                            f"📁 **出力先情報**\n\n"
                            f"| 項目 | パス |\n"
                            f"|---|---|\n"
                            f"| 📊 Excelファイル | `{output_xlsx}` |\n"
                            f"| 🖼️ プロット保存先 | `{dirs['plots']}` |\n"
                            f"| 📂 出力フォルダ | `{dirs['plots'].parent}` |"
                        )

                    except Exception as e:
                        st.error(f"出力エラー:\n{traceback.format_exc()}")

    # ----------------------------------------------------
    # TAB 3: タイムコース表示
    # ----------------------------------------------------
    with tab3:
        st.header("タイムコース反応表示")
        profile_df = st.session_state["profile_df"]
        df = st.session_state["df"]

        if profile_df is not None and df is not None:
            items = list(profile_df["項目名"].unique())
            if not items:
                st.warning("プロファイルデータに項目名がありません。")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    tc_item = st.selectbox("表示項目", options=items)

                # Dynamic range for the selected item
                vmin, vmax = 0.0, 1000.0
                if tc_item in df.columns:
                    vals = pd.to_numeric(df[tc_item], errors="coerce").dropna()
                    if not vals.empty:
                        vmin, vmax = float(vals.min()), float(vals.max())

                with col2:
                    tc_conc_range = st.slider("濃度範囲", min_value=float(vals.min()) if not vals.empty else 0.0,
                                              max_value=float(vals.max()) if not vals.empty else 1000.0,
                                              value=(vmin, vmax), step=0.1)

                times = sorted(profile_df["時間"].unique())
                baseline_time_options = {"(生データ表示)": None}
                baseline_time_options.update({f"{t:.1f}s": t for t in times})

                col3, col4 = st.columns(2)
                with col3:
                    tc_outlier = st.selectbox("乖離選択（一般検体のみ適用）", options=["all", "outlier", "normal"],
                                              format_func=lambda x: {"all":"全て", "outlier":"乖離のみ", "normal":"非乖離のみ"}[x])
                with col4:
                    tc_baseline_time_name = st.selectbox("基準時間(秒)", options=list(baseline_time_options.keys()))
                    tc_baseline_time = baseline_time_options[tc_baseline_time_name]

                col5, col6 = st.columns(2)
                with col5:
                    show_calibrators = st.checkbox("キャリブレーター表示", value=True,
                                                   help="C+数字パターン（例: C001）の検体を表示します。緑色で描画されます。")
                with col6:
                    st.markdown("""<div style='padding-top:8px; font-size:0.85em; color:#666;'>
                        🟢 キャリブレーターは乖離フィルタの対象外です
                    </div>""", unsafe_allow_html=True)

                if st.button("タイムコース表示", type="primary"):
                    metadata = st.session_state.get("metadata_enhanced") or st.session_state.get("metadata")
                    ref_outlier_map = st.session_state.get("ref_outlier_map")
                    id_col = st.session_state.get("id_col")

                    cmin, cmax = tc_conc_range
                    df_filtered = df[(pd.to_numeric(df[tc_item], errors='coerce') >= cmin) &
                                     (pd.to_numeric(df[tc_item], errors='coerce') <= cmax)]
                    allowed_sids = set(df_filtered[id_col].astype(str).tolist())

                    id_mapping = {}
                    if "依頼No." in df.columns and id_col in df.columns:
                        id_mapping = dict(zip(df["依頼No."].astype(str), df[id_col].astype(str)))

                    df_item = profile_df[profile_df["項目名"] == tc_item]
                    fig, ax = plt.subplots(figsize=(12, 7))

                    from matplotlib.lines import Line2D

                    plotted_count = 0
                    calib_count = 0
                    for sid, gdf in df_item.groupby("依頼No."):
                        sid_str = str(sid)
                        mapped_id = id_mapping.get(sid_str, sid_str)
                        if mapped_id not in allowed_sids: continue

                        calib = is_calibrator(sid_str)

                        # ── キャリブレーター ──────────────────────────────────
                        if calib:
                            if not show_calibrators:
                                continue
                            # キャリブレーターは乖離判定と独立した固有カテゴリ
                            color, lw, alpha = "#2ca02c", 1.5, 0.55  # 緑
                            calib_count += 1

                        # ── 一般検体 ─────────────────────────────────────────
                        else:
                            level = "none"
                            if ref_outlier_map and mapped_id in ref_outlier_map:
                                level = ref_outlier_map[mapped_id]
                            elif metadata and mapped_id in metadata and "outliers" in metadata[mapped_id]:
                                levels = [v["level"] for v in metadata[mapped_id]["outliers"].values()]
                                if "strong_candidate" in levels: level = "strong_candidate"
                                elif "candidate" in levels: level = "candidate"
                                elif "mild_candidate" in levels: level = "mild_candidate"

                            is_outlier = level in ["strong_candidate", "candidate", "mild_candidate"]
                            if tc_outlier == "outlier" and not is_outlier: continue
                            if tc_outlier == "normal" and is_outlier: continue

                            if level == "strong_candidate":   color, lw, alpha = "red",     1.5, 0.9
                            elif level == "candidate":        color, lw, alpha = "orange",  1.5, 0.8
                            elif level == "mild_candidate":   color, lw, alpha = "yellow",  1.5, 0.7
                            else:                             color, lw, alpha = "#1f77b4", 1.5, 0.3

                        time_vals = gdf["時間"].values
                        abs_vals = gdf["吸光度"].values

                        if tc_baseline_time is not None:
                            idx = np.argmin(np.abs(time_vals - tc_baseline_time))
                            base_abs = abs_vals[idx]
                            abs_vals = abs_vals - base_abs

                        ax.plot(time_vals, abs_vals, color=color, linewidth=lw, alpha=alpha)
                        plotted_count += 1

                    ax.set_title(
                        f"タイムコース反応: {tc_item}  "
                        f"（一般検体: {plotted_count - calib_count}件"
                        + (f", キャリブレーター: {calib_count}件" if show_calibrators else "")
                        + f", 基準時間: {tc_baseline_time if tc_baseline_time is not None else 'None'}s)"
                    )
                    ax.set_xlabel("時間(秒)")
                    ax.set_ylabel("吸光度")
                    ax.grid(True, alpha=0.3)
                    if tc_baseline_time is not None:
                        ax.axvline(tc_baseline_time, color='black', linestyle='--', alpha=0.5)

                    # 凡例
                    legend_elements = [
                        Line2D([0], [0], color="#1f77b4", lw=1.5, alpha=0.7, label="一般検体（正常）"),
                        Line2D([0], [0], color="yellow",  lw=1.5, alpha=0.9, label="一般検体（軽度乖離）"),
                        Line2D([0], [0], color="orange",  lw=1.5, alpha=0.9, label="一般検体（乖離）"),
                        Line2D([0], [0], color="red",     lw=1.5, alpha=0.9, label="一般検体（強乖離）"),
                    ]
                    if show_calibrators:
                        legend_elements.append(
                            Line2D([0], [0], color="#2ca02c", lw=1.5, alpha=0.8, label="キャリブレーター")
                        )
                    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
                              framealpha=0.85, edgecolor="#cccccc")

                    st.pyplot(fig)
                    plt.close(fig)
    # ----------------------------------------------------
    # TAB 4: タイムコース解析
    # ----------------------------------------------------
    with tab4:
        st.header("タイムコース解析（吸光度からの濃度再計算）")
        profile_df = st.session_state["profile_df"]
        measurement_df = st.session_state["df"]

        if profile_df is not None and measurement_df is not None:
            items = list(profile_df["項目名"].unique())
            if not items:
                st.warning("プロファイルデータに項目名がありません。")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    tc_item_tab4 = st.selectbox("解析項目", options=items, key="tc_item_tab4")

                # Detect calibrator groups
                cal_groups = find_calibrator_groups(measurement_df, tc_item_tab4)

                st.subheader("キャリブレーター設定")
                if cal_groups:
                    group_opts = {}
                    for i, g in enumerate(cal_groups):
                        label = f"グループ {i+1}: {g['ids'][0]} ~ {g['ids'][-1]} ({len(g['ids'])}点データ)"
                        group_opts[label] = g['ids']
                    group_opts["カスタム入力 (手動指定)"] = None

                    selected_group_label = st.selectbox(
                        "キャリブレーターグループ選択",
                        options=list(group_opts.keys()),
                        index=0
                    )
                    selected_ids = group_opts[selected_group_label]

                    if selected_ids is not None:
                        default_ids_str = ", ".join(selected_ids)
                    else:
                        default_ids_str = "C001, C002, C003, C004, C005, C006"
                else:
                    st.info("※ キャリブレーターグループが自動検出されなかったため、手動入力してください。")
                    default_ids_str = "C001, C002, C003, C004, C005, C006"

                col3, col4 = st.columns(2)
                with col3:
                    cal_ids_str = st.text_input("キャリブレーターID (カンマ区切り)", value=default_ids_str)
                with col4:
                    cal_concs_str = st.text_input("キャリブレーター濃度 (例: 0.0, 5.3, ...)", value="0.0, 5.3, 14.0, 30.4, 56.7, 139.8")

                # Points & Curve Modes
                st.subheader("測光ポイント & 検量線モード設定")
                st.info("計算式: 処理値(mAbs/min) = {(Abs_end - Abs_start) * 0.1} / {(Time_end - Time_start) / 60}")

                # Get available time points
                times = sorted(profile_df[profile_df["項目名"] == tc_item_tab4]["時間"].unique())
                time_opts = {f"{t:.1f}s": t for t in times}

                col5, col6 = st.columns(2)
                with col5:
                    st.write("**デフォルト設定 (Base)**")
                    base_start_name = st.selectbox("開始時間 (Base)", options=list(time_opts.keys()), index=min(9, len(time_opts)-1), key="base_start")
                    base_end_name = st.selectbox("終了時間 (Base)", options=list(time_opts.keys()), index=min(19, len(time_opts)-1), key="base_end")
                    curve_mode_base = st.selectbox("検量線モード (Base)", options=["piecewise_linear", "spline"], format_func=lambda x: "折れ線" if x == "piecewise_linear" else "スプライン", key="curve_mode_base")
                with col6:
                    st.write("**変更後設定 (New)**")
                    new_start_name = st.selectbox("開始時間 (New)", options=list(time_opts.keys()), index=min(14, len(time_opts)-1), key="new_start")
                    new_end_name = st.selectbox("終了時間 (New)", options=list(time_opts.keys()), index=min(24, len(time_opts)-1), key="new_end")
                    curve_mode_new = st.selectbox("検量線モード (New)", options=["piecewise_linear", "spline"], format_func=lambda x: "折れ線" if x == "piecewise_linear" else "スプライン", key="curve_mode_new")

                show_tc_outliers_tab4 = st.checkbox(
                    "プロット図で乖離・非乖離を色分け表示",
                    value=True,
                    help="相関解析で判定された乖離レベルに基づき、New予測濃度のサンプル点を強乖離(赤)、乖離(オレンジ)、軽度乖離(黄)、非乖離(青)に色分け表示します。"
                )

                if st.button("2. 解析実行", type="primary", key="btn_tc_analyze"):
                    cal_ids = [s.strip() for s in cal_ids_str.split(",") if s.strip()]
                    cal_concs = []
                    for s in cal_concs_str.split(","):
                        if s.strip():
                            try:
                                cal_concs.append(float(s.strip()))
                            except ValueError:
                                pass

                    if len(cal_ids) != len(cal_concs):
                        st.error(f"キャリブレーターIDの数 ({len(cal_ids)}件) と濃度の数 ({len(cal_concs)}件) が一致しません。")
                    else:
                        with st.spinner("解析中..."):
                            import numpy as np
                            from scipy.interpolate import interp1d

                            df_item = profile_df[profile_df["項目名"] == tc_item_tab4]

                            def calc_rate(time_start, time_end):
                                rates = {}
                                for sid, gdf in df_item.groupby("依頼No."):
                                    gdf_sorted = gdf.sort_values("時間")
                                    t_vals = gdf_sorted["時間"].values
                                    a_vals = gdf_sorted["吸光度"].values

                                    idx_s = np.argmin(np.abs(t_vals - time_start))
                                    idx_e = np.argmin(np.abs(t_vals - time_end))

                                    t_s, a_s = t_vals[idx_s], a_vals[idx_s]
                                    t_e, a_e = t_vals[idx_e], a_vals[idx_e]

                                    if t_e != t_s:
                                        rate = ((a_e - a_s) * 0.1) / ((t_e - t_s) / 60.0)
                                        rates[str(sid)] = rate
                                return rates

                            base_t_s = time_opts[base_start_name]
                            base_t_e = time_opts[base_end_name]
                            new_t_s = time_opts[new_start_name]
                            new_t_e = time_opts[new_end_name]

                            base_rates = calc_rate(base_t_s, base_t_e)
                            new_rates = calc_rate(new_t_s, new_t_e)

                            cal_base_rates = [base_rates.get(cid, np.nan) for cid in cal_ids]
                            cal_new_rates = [new_rates.get(cid, np.nan) for cid in cal_ids]

                            # Filter out missing cals for New and Base
                            valid_new_cals = [(r_new, c) for r_new, c in zip(cal_new_rates, cal_concs) if not np.isnan(r_new)]
                            valid_base_cals = [(r_base, c) for r_base, c in zip(cal_base_rates, cal_concs) if not np.isnan(r_base)]

                            if len(valid_new_cals) < 2 or len(valid_base_cals) < 2:
                                st.error("有効なキャリブレーターデータが不足しています。")
                            else:
                                valid_new_cals.sort(key=lambda x: x[0])
                                x_cals_new = np.array([x[0] for x in valid_new_cals])
                                y_cals_new = np.array([x[1] for x in valid_new_cals])

                                valid_base_cals.sort(key=lambda x: x[0])
                                x_cals_base = np.array([x[0] for x in valid_base_cals])
                                y_cals_base = np.array([x[1] for x in valid_base_cals])

                                def predict_new(x_val):
                                    if np.isnan(x_val): return np.nan
                                    if curve_mode_new == "piecewise_linear":
                                        return float(np.interp(x_val, x_cals_new, y_cals_new))
                                    else:
                                        f = interp1d(x_cals_new, y_cals_new, kind='cubic', fill_value="extrapolate")
                                        return float(f(x_val))

                                def predict_base(x_val):
                                    if np.isnan(x_val): return np.nan
                                    if curve_mode_base == "piecewise_linear":
                                        return float(np.interp(x_val, x_cals_base, y_cals_base))
                                    else:
                                        f = interp1d(x_cals_base, y_cals_base, kind='cubic', fill_value="extrapolate")
                                        return float(f(x_val))

                                cal_id_to_conc = dict(zip(cal_ids, cal_concs))

                                # 乖離レベル判定準備
                                ref_outlier_map = st.session_state.get("ref_outlier_map")
                                metadata_session = st.session_state.get("metadata_enhanced") or st.session_state.get("metadata")
                                id_col_session = st.session_state.get("id_col", "SampleID")
                                id_mapping = {}
                                if id_col_session and id_col_session != "依頼No." and id_col_session in measurement_df.columns:
                                    for _, r_row in measurement_df.iterrows():
                                        rid_str = str(r_row.get("依頼No.", ""))
                                        sid_str = str(r_row.get(id_col_session, ""))
                                        if rid_str and sid_str:
                                            id_mapping[rid_str] = sid_str

                                def get_sample_outlier_level(sid_raw):
                                    sid_str = str(sid_raw)
                                    mapped_id = id_mapping.get(sid_str, sid_str)
                                    if ref_outlier_map and mapped_id in ref_outlier_map:
                                        return ref_outlier_map[mapped_id]
                                    if metadata_session and mapped_id in metadata_session and "outliers" in metadata_session[mapped_id]:
                                        levels = [v["level"] for v in metadata_session[mapped_id]["outliers"].values()]
                                        if "strong_candidate" in levels: return "strong_candidate"
                                        if "candidate" in levels: return "candidate"
                                        if "mild_candidate" in levels: return "mild_candidate"
                                    return "none"

                                level_label_map = {
                                    "strong_candidate": "強乖離",
                                    "candidate": "乖離",
                                    "mild_candidate": "軽度乖離",
                                    "none": "非乖離"
                                }

                                results = []
                                for sid in base_rates.keys():
                                    b_r = base_rates[sid]
                                    n_r = new_rates[sid]
                                    
                                    if is_calibrator(sid):
                                        orig_conc = cal_id_to_conc.get(sid, np.nan)
                                        outlier_status = "キャリブレーター"
                                    else:
                                        m_row = measurement_df[measurement_df["依頼No."].astype(str) == sid]
                                        if not m_row.empty:
                                            orig_conc = pd.to_numeric(m_row.iloc[0].get(tc_item_tab4, np.nan), errors='coerce')
                                        else:
                                            orig_conc = np.nan
                                        outlier_status = level_label_map.get(get_sample_outlier_level(sid), "非乖離")

                                    # Base検量線で再計算した濃度（プロット用）
                                    base_recalc_conc = predict_base(b_r)
                                    new_conc = predict_new(n_r)
                                    results.append({
                                        "依頼No.": sid,
                                        "乖離判定": outlier_status,
                                        "元の処理値 (Base)": b_r,
                                        "新たな処理値 (New)": n_r,
                                        "装置測定値": orig_conc,
                                        "Base再計算濃度": base_recalc_conc,
                                        "New予測濃度": new_conc,
                                        "濃度差 (New - Base)": new_conc - base_recalc_conc if not np.isnan(base_recalc_conc) else np.nan
                                    })

                                res_df = pd.DataFrame(results)

                                # 単位の取得
                                unit = ""
                                parsed_dir = st.session_state.get("parsed_dir")
                                if parsed_dir and (parsed_dir / "metadata.json").exists():
                                    try:
                                        import json
                                        with open(parsed_dir / "metadata.json", "r", encoding="utf-8") as f:
                                            disk_metadata = json.load(f)
                                        if "measurement_units" in disk_metadata:
                                            unit_dict = disk_metadata["measurement_units"]
                                            if tc_item_tab4 in unit_dict:
                                                unit = f" ({unit_dict[tc_item_tab4]})"
                                    except Exception as e:
                                        st.warning(f"metadata.jsonの読み込みに失敗しました: {e}")
                                elif st.session_state.get("metadata") and "measurement_units" in st.session_state["metadata"]:
                                    unit_dict = st.session_state["metadata"]["measurement_units"]
                                    if tc_item_tab4 in unit_dict:
                                        unit = f" ({unit_dict[tc_item_tab4]})"

                                # Valid base cals for plotting (sorted by conc)
                                valid_base_cals_plot = [(r_base, c) for r_base, c in zip(cal_base_rates, cal_concs) if not np.isnan(r_base)]
                                valid_base_cals_plot.sort(key=lambda x: x[1])
                                x_base_cals_plot = np.array([x[1] for x in valid_base_cals_plot]) # Conc
                                y_base_cals_plot = np.array([x[0] for x in valid_base_cals_plot]) # Rate

                                # Valid new cals for plotting (sorted by conc)
                                valid_new_cals_plot = [(r_new, c) for r_new, c in zip(cal_new_rates, cal_concs) if not np.isnan(r_new)]
                                valid_new_cals_plot.sort(key=lambda x: x[1])
                                x_new_cals_plot = np.array([x[1] for x in valid_new_cals_plot]) # Conc
                                y_new_cals_plot = np.array([x[0] for x in valid_new_cals_plot]) # Rate

                                st.subheader("解析結果")

                                mode_base_str = "折れ線" if curve_mode_base == "piecewise_linear" else "スプライン"
                                mode_new_str = "折れ線" if curve_mode_new == "piecewise_linear" else "スプライン"

                                # Plot Calibration Curve
                                fig, ax = plt.subplots(figsize=(10, 6))

                                # --- Calibrator markers (scatter) ---
                                ax.scatter(x_base_cals_plot, y_base_cals_plot, color='gray', marker='s',
                                           s=80, zorder=5, label="Calibrators (Base)")
                                ax.scatter(x_new_cals_plot, y_new_cals_plot, color='black', marker='o',
                                           s=40, zorder=6, label="Calibrators (New)")

                                # --- Base calibration curve ---
                                if curve_mode_base == "piecewise_linear":
                                    ax.plot(x_base_cals_plot, y_base_cals_plot, color='gray',
                                            linestyle='-', linewidth=2, alpha=0.8, zorder=3,
                                            label=f"Cal Curve (Base: {mode_base_str})")
                                else:
                                    rate_dense_base = np.linspace(y_base_cals_plot.min(), y_base_cals_plot.max(), 200)
                                    conc_dense_base = [predict_base(r) for r in rate_dense_base]
                                    ax.plot(conc_dense_base, rate_dense_base, color='gray',
                                            linestyle='--', linewidth=2, alpha=0.8, zorder=3,
                                            label=f"Cal Curve (Base: {mode_base_str})")

                                # --- New calibration curve ---
                                if curve_mode_new == "piecewise_linear":
                                    ax.plot(x_new_cals_plot, y_new_cals_plot, color='red',
                                            linestyle='-', linewidth=1.5, alpha=0.8, zorder=4,
                                            label=f"Cal Curve (New: {mode_new_str})")
                                else:
                                    rate_dense_new = np.linspace(y_new_cals_plot.min(), y_new_cals_plot.max(), 200)
                                    conc_dense_new = [predict_new(r) for r in rate_dense_new]
                                    ax.plot(conc_dense_new, rate_dense_new, color='red',
                                            linestyle='--', linewidth=1.5, alpha=0.8, zorder=4,
                                            label=f"Cal Curve (New: {mode_new_str})")

                                # --- Plot samples (Base) - predict_base で再計算した濃度を使用 ---
                                sample_x_base = [r["Base再計算濃度"] for r in results if not is_calibrator(r["依頼No."]) and not np.isnan(r["Base再計算濃度"])]
                                sample_y_base = [r["元の処理値 (Base)"] for r in results if not is_calibrator(r["依頼No."]) and not np.isnan(r["Base再計算濃度"])]
                                if sample_x_base:
                                    ax.scatter(sample_x_base, sample_y_base, color='gray', alpha=0.4,
                                               zorder=7, label="Samples (Base)")

                                # --- Plot samples (New) ---
                                if show_tc_outliers_tab4:
                                    level_plot_specs = [
                                        ("none", "blue", "Samples New (非乖離)", 8, "o"),
                                        ("mild_candidate", "gold", "Samples New (軽度乖離)", 9, "^"),
                                        ("candidate", "orange", "Samples New (乖離)", 9, "^"),
                                        ("strong_candidate", "red", "Samples New (強乖離)", 9, "^"),
                                    ]
                                    for lvl_key, col_val, lbl_val, z_val, mkr in level_plot_specs:
                                        sub_x = [r["New予測濃度"] for r in results if not is_calibrator(r["依頼No."]) and not np.isnan(r["New予測濃度"]) and get_sample_outlier_level(r["依頼No."]) == lvl_key]
                                        sub_y = [r["新たな処理値 (New)"] for r in results if not is_calibrator(r["依頼No."]) and not np.isnan(r["New予測濃度"]) and get_sample_outlier_level(r["依頼No."]) == lvl_key]
                                        if sub_x:
                                            ax.scatter(sub_x, sub_y, color=col_val, marker=mkr, alpha=0.75,
                                                       zorder=z_val, label=lbl_val)
                                else:
                                    sample_x_new = [r["New予測濃度"] for r in results if not is_calibrator(r["依頼No."]) and not np.isnan(r["New予測濃度"])]
                                    sample_y_new = [r["新たな処理値 (New)"] for r in results if not is_calibrator(r["依頼No."]) and not np.isnan(r["New予測濃度"])]
                                    if sample_x_new:
                                        ax.scatter(sample_x_new, sample_y_new, color='blue', alpha=0.6,
                                                   zorder=8, label="Samples (New)")

                                ax.set_xlabel(f"濃度{unit}")
                                ax.set_ylabel("処理値 (mAbs/min)")
                                ax.set_title(f"検量線 (Base: {mode_base_str}, New: {mode_new_str})")
                                ax.grid(True, alpha=0.3)
                                ax.legend(fontsize=8, loc='best')
                                st.pyplot(fig)
                                plt.close(fig)

                                st.dataframe(res_df)

