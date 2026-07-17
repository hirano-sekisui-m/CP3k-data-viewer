import sys
from pathlib import Path
import json
import traceback
import itertools
import math

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
    if "SampleID" not in measurement_df.columns:
        if "SID" in measurement_df.columns:
            measurement_df["SampleID"] = measurement_df["SID"].astype(str)
        elif "依頼No." in measurement_df.columns:
            measurement_df["SampleID"] = measurement_df["依頼No."].astype(str)
        else:
            measurement_df["SampleID"] = measurement_df.index.astype(str)

    id_col = pick_col(measurement_df, ID_COL_CANDIDATES, default="SID")
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

    tab1, tab2, tab3 = st.tabs(["相関解析", "Excel出力", "タイムコース表示"])

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
            z_thresh_val = st.slider("乖離z(MAD)", min_value=1.5, max_value=8.0, value=3.5, step=0.1)
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
            max_show_val = st.slider("表示上限", min_value=0, max_value=30, value=6, step=1)

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
                    sub_ref = df_ref[[ref_x, ref_y]].dropna()
                    if len(sub_ref) >= 2:
                        xr, yr = sub_ref[ref_x].astype(float).values, sub_ref[ref_y].astype(float).values
                        ar, br, _ = regression_fit_info(xr, yr, method=reg_method, deming_lambda=lam)
                        metrics_ref = compute_pair_sample_metrics(df_ref, id_col, group_col, ref_x, ref_y, ar, br, z_thresh=z_thresh)
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
                            metrics_df = compute_pair_sample_metrics(df_pair, id_col, group_col, xcol, ycol, a, b, z_thresh=z_thresh)

                            color_list = []
                            for _, row in metrics_df.iterrows():
                                sid = str(row[id_col])
                                sample_meta = run_metadata.setdefault(sid, {})
                                outliers_meta = sample_meta.setdefault("outliers", {})

                                z_mad = row.get("z_MAD", np.nan)
                                level = classify_outlier_level(abs(z_mad), thresh=z_thresh) if np.isfinite(z_mad) else "none"
                                outliers_meta[pair_key] = {"level": level, "z_MAD": float(z_mad) if np.isfinite(z_mad) else None}

                                target_level = ref_outlier_map.get(sid, "none") if ref_outlier_map else level

                                if target_level == "strong_candidate": color_list.append("red")
                                elif target_level == "candidate": color_list.append("orange")
                                elif target_level == "mild_candidate": color_list.append("yellow")
                                else: color_list.append("#1f77b4")

                            fig, used_sub, flagged, bias, loa = plot_suite(
                            df=df_pair, id_col=id_col, group_col=group_col, xcol=xcol, ycol=ycol,
                            method=method, lam=lam, a=a, b=b, r=r, fit_info=fit_info,
                            z_thresh=z_thresh, outlier_label_top=label_top_val,
                            fig_width=16, fig_height=10, dpi=100, external_colors=color_list
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

                st.session_state["analysis_results"] = {
                    "run_metadata": run_metadata,
                    "ref_outlier_map": ref_outlier_map,
                    "summary_rows": summary_rows,
                    "sample_metric_tables": sample_metric_tables,
                    "outlier_tables": outlier_tables,
                    "figures": figures
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
            if st.button("③Excel出力", type="primary"):
                with st.spinner("Excelファイル作成中..."):
                    try:
                        res = st.session_state["analysis_results"]
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

                        st.success(f"Excel保存完了! 出力先: {output_xlsx}")
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
                    tc_outlier = st.selectbox("乖離選択", options=["all", "outlier", "normal"],
                                              format_func=lambda x: {"all":"全て", "outlier":"乖離のみ", "normal":"非乖離のみ"}[x])
                with col4:
                    tc_baseline_time_name = st.selectbox("基準時間(秒)", options=list(baseline_time_options.keys()))
                    tc_baseline_time = baseline_time_options[tc_baseline_time_name]

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

                    plotted_count = 0
                    for sid, gdf in df_item.groupby("依頼No."):
                        sid_str = str(sid)
                        mapped_id = id_mapping.get(sid_str, sid_str)
                        if mapped_id not in allowed_sids: continue

                        color, lw, alpha = "#1f77b4", 1.0, 0.3
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

                        if level == "strong_candidate": color, lw, alpha = "red", 2.5, 0.9
                        elif level == "candidate": color, lw, alpha = "orange", 2.0, 0.8
                        elif level == "mild_candidate": color, lw, alpha = "yellow", 1.5, 0.7

                        time_vals = gdf["時間"].values
                        abs_vals = gdf["吸光度"].values

                        if tc_baseline_time is not None:
                            idx = np.argmin(np.abs(time_vals - tc_baseline_time))
                            base_abs = abs_vals[idx]
                            abs_vals = abs_vals - base_abs

                        ax.plot(time_vals, abs_vals, color=color, linewidth=lw, alpha=alpha)
                        plotted_count += 1

                    ax.set_title(f"タイムコース反応: {tc_item} (プロット数: {plotted_count}, 基準時間: {tc_baseline_time if tc_baseline_time is not None else 'None'}s)")
                    ax.set_xlabel("時間(秒)")
                    ax.set_ylabel("吸光度")
                    ax.grid(True, alpha=0.3)
                    if tc_baseline_time is not None: ax.axvline(tc_baseline_time, color='black', linestyle='--', alpha=0.5)
                    st.pyplot(fig)
                    plt.close(fig)
