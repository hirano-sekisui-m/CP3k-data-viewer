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

    csv_files = list(
        data_dir.glob("*.csv")
    )

    if len(csv_files) == 0:
        raise FileNotFoundError(
            f"No CSV found: {data_dir}"
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
            and "SID" in line
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
            and "項目名" in lines[i]
        ):
            profile_header_idx = i
            break

    measurement_lines = lines[
        header_idx:profile_header_idx - 2
    ]

    df = pd.read_csv(
        StringIO("".join(measurement_lines)),
        dtype=str
    )

    original_columns = list(df.columns)

    fixed_cols = original_columns[:5]

    measurement_items = []
    measurement_units = {}

    renamed_columns = fixed_cols.copy()

    idx = 5

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
            and "項目名" in line
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
            except:
                pass

        absorb_values = []

        for a in absorb_tokens:

            try:
                absorb_values.append(
                    float(a)
                )
            except:
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