from pathlib import Path
from datetime import datetime
import pandas as pd
import json
import re
import shutil
from io import StringIO


# ==========================================================
# Public API
# ==========================================================

def process_csv(csv_path):
    """
    生CSVを読み込み、
    measurement.parquet
    profile.parquet
    metadata.json
    を生成する。

    処理成功後は元CSVを
    raw-data/processed/
    に移動する。

    Returns
    -------
    measurement_df
    profile_df
    metadata
    """

    csv_path = Path(csv_path)

    with open(
        csv_path,
        "r",
        encoding="cp932",
        errors="replace"
    ) as f:
        lines = f.readlines()

    measurement_df, metadata = _parse_measurement_table(
        lines
    )

    profile_df, item_mapping = _parse_profile_table(
        lines
    )

    metadata["item_mapping"] = item_mapping

    _save_outputs(
        csv_path,
        measurement_df,
        profile_df,
        metadata
    )

    _move_original_csv(csv_path)

    return (
        measurement_df,
        profile_df,
        metadata
    )


def load_parsed_data(parsed_dir):
    """
    parsed-data配下から再読込
    """

    parsed_dir = Path(parsed_dir)

    measurement_df = pd.read_parquet(
        parsed_dir / "measurement.parquet"
    )

    profile_df = pd.read_parquet(
        parsed_dir / "profile.parquet"
    )

    with open(
        parsed_dir / "metadata.json",
        encoding="utf-8"
    ) as f:
        metadata = json.load(f)

    return (
        measurement_df,
        profile_df,
        metadata
    )


def find_latest_csv(
    data_dir="../../data/raw-data"
):
    data_dir = Path(data_dir)

    if not data_dir.is_absolute():
        data_dir = (
            Path(__file__).resolve().parents[2]
            / data_dir
        )

    csv_files = list(
        data_dir.glob("*.csv")
    )

    if len(csv_files) == 0:
        raise FileNotFoundError(
            f"No CSV found: {data_dir}. "
            f"Place a .csv file under the raw-data directory first."
        )

    csv_files.sort(
        key=lambda x: x.stat().st_mtime
    )

    return csv_files[-1]


def get_measurement_items(metadata):

    return metadata[
        "measurement_items"
    ]


def get_unit(metadata, item_name):

    return (
        metadata
        .get("measurement_units", {})
        .get(item_name, "")
    )


# ==========================================================
# Measurement
# ==========================================================

def _parse_measurement_table(lines):

    header_idx = None

    for i, line in enumerate(lines):

        if (
            "依頼No." in line
            and ("PID" in line or "SID" in line or "測定日" in line or "属性" in line)
        ):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(
            "Measurement header not found."
        )

    profile_header_idx = None

    for i in range(header_idx + 1, len(lines)):

        if (
            "依頼No." in lines[i]
            and ("項目名" in lines[i] or "測光" in lines[i] or "ﾌﾟﾛﾌｧｲﾙ" in lines[i])
        ):
            profile_header_idx = i
            break

    if profile_header_idx is not None:
        # プロファイルヘッダー直前の空行までを測定値領域とする
        end_idx = profile_header_idx
        while end_idx > header_idx and not lines[end_idx - 1].strip():
            end_idx -= 1
        measurement_lines = lines[header_idx:end_idx]
    else:
        measurement_lines = lines[header_idx:]

    df = pd.read_csv(
        StringIO("".join(measurement_lines)),
        dtype=str
    )

    original_columns = list(df.columns)

    # 固定列は 4列 ('依頼No.', 'PID'/'SID', '測定日', '属性')
    num_fixed = 4
    fixed_cols = original_columns[:num_fixed]

    measurement_items = []
    measurement_units = {}

    renamed_columns = fixed_cols.copy()

    idx = num_fixed

    while idx < len(original_columns):

        raw_name = str(
            original_columns[idx]
        ).strip()

        match = re.match(
            r"^(.*?)\((.*?)\)$",
            raw_name
        )

        if match:

            item_name = match.group(1).strip()

            unit = match.group(2).strip()

            measurement_units[
                item_name
            ] = unit

        else:

            item_name = raw_name

        measurement_items.append(
            item_name
        )

        renamed_columns.append(
            item_name
        )

        if idx + 1 < len(original_columns):

            renamed_columns.append(
                f"{item_name}_FLAG"
            )

        idx += 2

    df.columns = renamed_columns[
        :len(df.columns)
    ]

    for col in df.columns:

        if col.endswith("_FLAG"):

            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace("nan", "")
            )

    fixed = {
        "依頼No.",
        "SID",
        "検体ﾊﾞｰｺｰﾄﾞ",
        "測定日",
        "属性"
    }

    for col in df.columns:

        if col in fixed:
            continue

        if col.endswith("_FLAG"):
            continue

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    metadata = {
        "measurement_items":
            measurement_items,

        "measurement_units":
            measurement_units
    }

    return df, metadata


# ==========================================================
# Profile
# ==========================================================

def _parse_profile_table(lines):

    profile_header_idx = None

    for i, line in enumerate(lines):

        if (
            "依頼No." in line
            and ("項目名" in line or "測光" in line or "ﾌﾟﾛﾌｧｲﾙ" in line)
        ):
            profile_header_idx = i
            break

    if profile_header_idx is None:
        raise ValueError(
            "Profile header not found."
        )

    records = []

    item_mapping = {}

    i = profile_header_idx + 1

    while i < len(lines):

        row1 = lines[i].strip()

        if row1 == "":
            i += 1
            continue

        if i + 1 >= len(lines):
            break

        row2 = lines[i + 1].strip()

        token1 = [
            x.replace('"', '').strip()
            for x in row1.split(",")
        ]

        token2 = [
            x.replace('"', '').strip()
            for x in row2.split(",")
        ]

        if len(token1) < 10:

            i += 1
            continue

        request_no = token1[0]

        item_name = token1[1]

        item_no = pd.to_numeric(
            token1[2],
            errors="coerce"
        )

        photometric_port = pd.to_numeric(
            token1[3],
            errors="coerce"
        )

        processed_value = pd.to_numeric(
            token1[4],
            errors="coerce"
        )

        if not pd.isna(item_no):
            item_mapping[
                str(int(item_no))
            ] = item_name

        time_tokens = token1[6:]

        absorb_tokens = token2[6:]

        time_values = []

        for t in time_tokens:

            try:
                time_values.append(
                    float(t)
                )
            except Exception:
                pass

        absorb_values = []

        for a in absorb_tokens:

            try:
                absorb_values.append(
                    float(a)
                )
            except Exception:
                pass

        n = min(
            len(time_values),
            len(absorb_values)
        )

        for j in range(n):

            records.append({
                "依頼No.": request_no,
                "項目名": item_name,
                "項目No.": item_no,
                "測光ﾎﾟｰﾄ": photometric_port,
                "処理値": processed_value,
                "時間": time_values[j],
                "吸光度": absorb_values[j]
            })

        i += 3

    profile_df = pd.DataFrame(
        records
    )

    return (
        profile_df,
        item_mapping
    )


# ==========================================================
# Save Outputs
# ==========================================================

def _save_outputs(
    csv_path,
    measurement_df,
    profile_df,
    metadata
):

    project_root = (
        csv_path.parent.parent.parent
    )

    parsed_root = (
        project_root
        / "data"
        / "parsed-data"
    )

    csv_stem = csv_path.stem

    output_dir = (
        parsed_root
        / csv_stem
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    measurement_df.to_parquet(
        output_dir
        / "measurement.parquet",
        index=False
    )

    profile_df.to_parquet(
        output_dir
        / "profile.parquet",
        index=False
    )

    metadata["source_csv"] = (
        csv_path.name
    )

    with open(
        output_dir / "metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2
        )


# ==========================================================
# Move Original CSV
# ==========================================================

def _move_original_csv(csv_path):

    processed_dir = (
        csv_path.parent
        / "processed"
    )

    processed_dir.mkdir(
        exist_ok=True
    )

    timestamp = (
        datetime.now()
        .strftime("%Y%m%d_%H%M%S")
    )

    dest = (
        processed_dir
        / f"{timestamp}_{csv_path.name}"
    )

    shutil.move(
        str(csv_path),
        str(dest)
    )

# ==========================================================
# TEST / ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    import sys

    project_root = Path(__file__).resolve().parents[2]

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    raw_data_dir = project_root / "data" / "raw-data"

    try:
        csv_file = find_latest_csv(raw_data_dir)
    except FileNotFoundError as exc:
        print(exc)
        raise SystemExit(1)

    print(f"CSV : {csv_file}")

    measurement_df, profile_df, metadata = process_csv(csv_file)

    print()
    print("measurement")
    print(measurement_df.shape)
    print()
    print("profile")
    print(profile_df.shape)
    print()
    print("metadata")
    print(list(metadata.keys()))
    print()
    print("Done")


# ==========================================================
# Parquet/metadata helper for analysis
# ==========================================================

def _try_parse_json_obj(val):
    """文字列化されたJSONや辞書オブジェクトから値を取り出すヘルパー"""

    if val is None:
        return None

    if isinstance(val, dict):
        return val

    if isinstance(val, str):

        s = val.strip()

        # JSONっぽければパースを試みる
        if s.startswith("{") and s.endswith("}"):
            try:
                return json.loads(s)
            except Exception:
                return None

        return s

    return None


def merge_raw_csv_files(csv_paths):
    """
    複数の raw F12 CSV ファイルを1つの F12 CSV形式の行リスト(lines)に結合する。
    
    Parameters
    ----------
    csv_paths : list of Path or str
        結合対象の CSV ファイルのパスのリスト
        
    Returns
    -------
    merged_lines : list of str
    """
    if not csv_paths:
        return []

    meas_header = None
    meas_rows = []
    profile_header = None
    profile_rows = []

    for path in csv_paths:
        with open(path, "r", encoding="cp932", errors="replace") as f:
            lines = f.readlines()

        if not lines:
            continue

        # 測定値ヘッダーとプロファイルヘッダーの位置を特定
        m_head_idx = None
        p_head_idx = None

        for i, line in enumerate(lines):
            stripped = line.strip()
            if m_head_idx is None and ("依頼No." in stripped or "PID" in stripped or "測定日" in stripped):
                m_head_idx = i
            elif p_head_idx is None and ("項目名" in stripped and ("測光" in stripped or "ﾌﾟﾛﾌｧｲﾙ" in stripped or "処理値" in stripped)):
                p_head_idx = i
                break

        if m_head_idx is None:
            m_head_idx = 0

        # セクション切り分け
        if p_head_idx is not None:
            raw_meas = lines[m_head_idx:p_head_idx]
            raw_prof = lines[p_head_idx:]
        else:
            raw_meas = lines[m_head_idx:]
            raw_prof = []

        # 測定値セクションの処理
        if raw_meas:
            if meas_header is None:
                meas_header = raw_meas[0]
            # データ行のみ抽出 (ヘッダーと空行を除く)
            for r in raw_meas[1:]:
                if r.strip() and not ("依頼No." in r and "PID" in r):
                    meas_rows.append(r)

        # プロファイルセクションの処理
        if raw_prof:
            if profile_header is None:
                profile_header = raw_prof[0]
            # データ行のみ抽出 (ヘッダーと空行を除く)
            for r in raw_prof[1:]:
                if r.strip() and not ("項目名" in r and "測光" in r):
                    profile_rows.append(r)

    # 統合した F12 CSV 行リストを作成
    merged_lines = []
    if meas_header:
        merged_lines.append(meas_header)
    merged_lines.extend(meas_rows)

    merged_lines.append("\n\n")

    if profile_header:
        merged_lines.append(profile_header)
    merged_lines.extend(profile_rows)

    return merged_lines


def process_multiple_csvs(csv_paths, output_dir=None):
    """
    複数の生CSVファイルをマージして1つの parsed ディレクトリ
    (measurement.parquet, profile.parquet, metadata.json) を生成する。
    """
    csv_paths = [Path(p) for p in csv_paths]
    if not csv_paths:
        raise ValueError("csv_paths が空です。")

    merged_lines = merge_raw_csv_files(csv_paths)

    measurement_df, metadata = _parse_measurement_table(merged_lines)
    profile_df, item_mapping = _parse_profile_table(merged_lines)
    metadata["item_mapping"] = item_mapping

    first_csv = csv_paths[0]
    if output_dir is None:
        parent_name = first_csv.parent.name
        parsed_root = first_csv.parents[2] / "parsed-data"
        output_dir = parsed_root / parent_name

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    measurement_df.to_parquet(output_dir / "measurement.parquet", index=False)
    profile_df.to_parquet(output_dir / "profile.parquet", index=False)

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return measurement_df, profile_df, metadata, output_dir


def extract_sample_id(measurement_df, preferred_keys=None):
    """
    `measurement.parquet` から検体IDを抽出して Series を返す。

    優先的に探すキーは日本語/英語の候補を順に試す。
    - measurement_df に直接 `SID`/`検体ID`/`PID`/`SampleID` などの列があればそれを使う
    - `属性` 列が辞書または JSON 文字列を含む場合は内部のキーを探す
    - 見つからなければ先頭列を ID として扱う

    戻り値は文字列 Series（index は元 DF に合わせる）
    """

    if preferred_keys is None:
        preferred_keys = [
            "SID",
            "検体ID",
            "PID",
            "SampleID",
            "ID",
            "検体ﾊﾞｰｺｰﾄﾞ",
            "属性",
            "依頼No."
        ]

    # 1) 直接存在する列を優先
    for key in preferred_keys:
        if key in measurement_df.columns:
            series = measurement_df[key]

            # attributes のように辞書/JSON を含む列なら内部から候補キーを探す
            if key == "属性":
                parsed = series.map(_try_parse_json_obj)

                # もし辞書が含まれていれば内部キーを探す
                if parsed.dropna().apply(lambda x: isinstance(x, dict)).any():
                    # 内部でよく使われるキーを試す
                    inner_keys = ["SID", "検体ID", "PID", "SampleID", "ID"]

                    for ik in inner_keys:
                        try:
                            extracted = parsed.map(lambda d: d.get(ik) if isinstance(d, dict) else None)
                        except Exception:
                            extracted = None

                        if extracted is not None and extracted.dropna().shape[0] > 0:
                            s = extracted.astype(str).str.strip()
                            if not s.isin(["None", "nan", "NaN", "", "<NA>"]).all():
                                return s

                # 辞書でなければそのまま文字列化
                s = series.astype(str).str.strip()
                if not s.isin(["None", "nan", "NaN", "", "<NA>"]).all():
                    return s
            else:
                s = series.astype(str).str.strip()
                if not s.isin(["None", "nan", "NaN", "", "<NA>"]).all():
                    return s

    # 2) 属性列がない場合、先頭列を ID として使う
    first_col = measurement_df.columns[0]
    return measurement_df[first_col].astype(str).str.strip()


def detect_prescription_columns(measurement_df, metadata):
    """
    `metadata.json` の `measurement_items` と `measurement_df` の列名の交差を返す。

    見つからなければ、列名が `処方` で始まる列を優先的に返す（後方互換）
    """

    items = metadata.get("measurement_items", []) if metadata else []

    # 交差
    common = [c for c in measurement_df.columns if c in items]

    if len(common) > 0:
        return common

    # フォールバック：列名が '処方' を含む/始まるもの
    fallback = [c for c in measurement_df.columns if str(c).startswith("処方") or "処方" in str(c)]

    return fallback


def load_parsed_for_analysis(parsed_dir, sample_id_col_name="SID"):
    """
    parsed ディレクトリ（例: data/parsed-data/260623_F12CSV_data）を読み込み、
    ・`measurement_df` に `sample_id_col_name` 列（デフォルト "SID"）を追加
    ・検出された処方列のリストを返す

    Returns
    -------
    measurement_df, profile_df, metadata, prescription_columns
    """

    measurement_df, profile_df, metadata = load_parsed_data(parsed_dir)

    sample_series = extract_sample_id(measurement_df)

    measurement_df = measurement_df.copy()
    measurement_df[sample_id_col_name] = sample_series.values

    pres_cols = detect_prescription_columns(measurement_df, metadata)

    return measurement_df, profile_df, metadata, pres_cols