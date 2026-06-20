import argparse
import datetime as dt
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from process_sensor import (
    data_timestamp_alignment,
    find_csv_by_keywords,
    interpolate_packet_timestamps,
)


DEVICE_KEYS = {
    "Headset": ["acc_headsetL", "gyro_headsetL", "quaternion_left"],
    "Phone": ["acc", "gyro", "linear_acc", "magnetic", "rotation"],
    "Watch": ["acc", "gyro", "line_acc", "mag", "ppg", "quaternion"],
    "STag_C63": ["acc", "gyro", "quaternion"],
    "STag_D4D": ["acc", "gyro", "quaternion"],
    "STag_": ["acc", "gyro", "quaternion"],
    "STag_2": ["acc", "gyro", "quaternion"],
}

INTERPOLATE_KEYS = {
    "Headset": ["acc_headsetL", "gyro_headsetL"],
    "Phone": [],
    "Watch": ["acc", "gyro", "line_acc", "mag", "ppg"],
    "STag_C63": ["acc", "gyro"],
    "STag_D4D": ["acc", "gyro"],
    "STag_": ["acc", "gyro"],
    "STag_2": ["acc", "gyro"],
}


def parse_timestamp(name):
    match = re.search(r"(\d{8})(\d{6})", name)
    if not match:
        return None
    return dt.datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")


def sorted_sequence_dirs(side_dir):
    if not side_dir.exists():
        return []
    dirs = []
    for path in side_dir.iterdir():
        if not path.is_dir():
            continue
        ts = parse_timestamp(path.name)
        if ts is not None:
            dirs.append((ts, path))
    return [path for _, path in sorted(dirs, key=lambda x: x[0])]


def pair_left_right(left_dirs, right_dirs, max_delta_sec):
    pairs = []
    for idx, left_dir in enumerate(left_dirs):
        right_dir = right_dirs[idx] if idx < len(right_dirs) else None
        delta = None
        status = "paired"
        if right_dir is None:
            status = "missing_right"
        else:
            left_ts = parse_timestamp(left_dir.name)
            right_ts = parse_timestamp(right_dir.name)
            delta = abs((right_ts - left_ts).total_seconds())
            if delta > max_delta_sec:
                status = "large_side_time_delta"
        pairs.append(
            {
                "index": idx + 1,
                "left": left_dir,
                "right": right_dir,
                "side_delta_seconds": delta,
                "status": status,
            }
        )
    return pairs


def available_device_names(seq_dir, side, no_stag):
    if seq_dir is None or not seq_dir.exists():
        return []
    names = {p.name for p in seq_dir.iterdir() if p.is_dir()}

    def has_real_csv(device_name):
        device_dir = seq_dir / device_name
        return sum(f.stat().st_size for f in device_dir.rglob("*.csv")) > 1024

    if side == "left":
        ordered = ["Headset", "Phone", "Watch"]
        if not no_stag:
            for tag_name in ["STag_C63", "STag_D4D", "STag_", "STag_2"]:
                if tag_name in names:
                    ordered.append(tag_name)
    else:
        # New phones may also contain Headset, but old downstream layout expects
        # right side to contribute only Phone and Watch.
        ordered = ["Phone", "Watch"]
    return [name for name in ordered if name in names and has_real_csv(name)]


def output_device_name(device_name, side):
    if device_name in ["STag_C63", "STag_"]:
        return "STag_left"
    if device_name in ["STag_D4D", "STag_2"]:
        return "STag_right"
    if device_name == "Headset":
        return "Headset"
    return f"{device_name}_{side}"


def load_csv(csv_file, key):
    if "rotation" in key.lower():
        df = pd.read_csv(csv_file, header=None, skiprows=1)
        df.columns = ["time", "w", "x", "y", "z"]
        return df
    return pd.read_csv(csv_file)


def align_device_modalities(device_dir, device_name):
    keys = DEVICE_KEYS[device_name]
    df_list = []
    time_col_list = []
    missing = []

    for key in keys:
        csv_file = find_csv_by_keywords(device_dir, [key])
        if csv_file is None:
            missing.append(key)
            continue

        df = load_csv(csv_file, key)
        time_cols = [col for col in df.columns if "time" in col.lower()]
        if not time_cols:
            missing.append(f"{key}:time")
            continue
        time_col = time_cols[0]

        if "baro" in key.lower():
            df[time_col] = (pd.to_numeric(df[time_col], errors="coerce") // 1000).astype(np.int64)

        if key in INTERPOLATE_KEYS[device_name]:
            df = interpolate_packet_timestamps(df, time_col)

        df_list.append((key, df))
        time_col_list.append(time_col)

    if missing:
        raise FileNotFoundError(f"{device_dir} missing keys: {missing}")

    min_time = max(df[time_col].min() for (_, df), time_col in zip(df_list, time_col_list))
    max_time = min(df[time_col].max() for (_, df), time_col in zip(df_list, time_col_list))
    cropped = []
    for (key, df), time_col in zip(df_list, time_col_list):
        data = df[(df[time_col] >= min_time) & (df[time_col] <= max_time)].reset_index(drop=True)
        if device_name != "Headset":
            data = data_timestamp_alignment(data)
        cropped.append((key, data))

    min_frames = min(len(df) for _, df in cropped)
    return [(key, df.iloc[:min_frames].reset_index(drop=True)) for key, df in cropped]


def process_sequence_side(seq_dir, side, save_seq_dir, no_stag):
    results = []
    for device_name in available_device_names(seq_dir, side, no_stag):
        device_dir = seq_dir / device_name
        aligned = align_device_modalities(str(device_dir), device_name)
        save_device_dir = save_seq_dir / output_device_name(device_name, side)
        save_device_dir.mkdir(parents=True, exist_ok=True)
        for key, df in aligned:
            original_csv = find_csv_by_keywords(str(device_dir), [key])
            save_path = save_device_dir / os.path.basename(original_csv)
            df.to_csv(save_path, index=False)
        results.append(
            {
                "side": side,
                "source_seq": seq_dir.name,
                "device": device_name,
                "output_device": save_device_dir.name,
                "frames": min(len(df) for _, df in aligned),
                "keys": [key for key, _ in aligned],
            }
        )
    return results


def copy_calibration_files(subject, calibration_root, output_seq_dirs):
    calib_dir = calibration_root / subject
    if not calib_dir.exists():
        return []
    copied = []
    def sort_key(path):
        match = re.match(r"^(\d+)(?:_(\d+))?_(\d{8})_(\d{6})\.pt$", path.name)
        if not match:
            return (999999, 999999, path.name)
        part = int(match.group(2)) if match.group(2) else 0
        return (int(match.group(1)), part, path.name)

    raw_calib_files = sorted(calib_dir.glob("*.pt"), key=sort_key)
    parsed = []
    for path in raw_calib_files:
        match = re.match(r"^(\d+)(?:_(\d+))?_(\d{8})_(\d{6})\.pt$", path.name)
        parsed.append((path, match))
    has_repeated_trials = any(match and match.group(2) for _, match in parsed)
    if has_repeated_trials:
        calib_files = raw_calib_files
    else:
        grouped = {}
        for path, match in parsed:
            if not match:
                grouped[path.name] = path
                continue
            seq = int(match.group(1))
            prev = grouped.get(seq)
            if prev is None or path.stat().st_size > prev.stat().st_size:
                grouped[seq] = path
        calib_files = [grouped[key] for key in sorted(grouped, key=lambda x: (999999, x) if isinstance(x, str) else (x, ""))]
    for idx, seq_dir in output_seq_dirs.items():
        if idx - 1 >= len(calib_files):
            break
        src = calib_files[idx - 1]
        dst = seq_dir / src.name
        shutil.copy2(src, dst)
        copied.append({"seq": idx, "file": src.name})
    return copied


def main():
    parser = argparse.ArgumentParser(description="Process raw SmartWear CSV folders into MocapDB raw/sensor layout.")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--input-subdir", default="sensor_raw")
    parser.add_argument("--output-subdir", default="sensor")
    parser.add_argument("--calibration-root", default="data/raw/calibration")
    parser.add_argument("--no-stag", action="store_true")
    parser.add_argument("--max-side-delta-sec", type=float, default=5.0)
    parser.add_argument("--copy-calibration", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    subject_input = raw_root / args.input_subdir / args.subject
    left_dirs = sorted_sequence_dirs(subject_input / "left")
    right_dirs = sorted_sequence_dirs(subject_input / "right")
    if not left_dirs:
        raise FileNotFoundError(f"No left sequence folders found under {subject_input / 'left'}")
    if not right_dirs:
        raise FileNotFoundError(f"No right sequence folders found under {subject_input / 'right'}")

    output_subject = raw_root / args.output_subdir / args.subject
    if output_subject.exists() and args.overwrite:
        shutil.rmtree(output_subject)
    output_subject.mkdir(parents=True, exist_ok=True)

    manifest = {
        "subject": args.subject,
        "no_stag": args.no_stag,
        "left_count": len(left_dirs),
        "right_count": len(right_dirs),
        "sequences": [],
    }

    output_seq_dirs = {}
    for pair in pair_left_right(left_dirs, right_dirs, args.max_side_delta_sec):
        seq_name = f"seq_{pair['index']:02d}"
        save_seq_dir = output_subject / seq_name
        if save_seq_dir.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists, use --overwrite: {save_seq_dir}")
        save_seq_dir.mkdir(parents=True, exist_ok=True)
        output_seq_dirs[pair["index"]] = save_seq_dir

        print(f"Processing {seq_name}: left={pair['left'].name}, right={pair['right'].name}")
        devices = []
        devices.extend(process_sequence_side(pair["left"], "left", save_seq_dir, args.no_stag))
        devices.extend(process_sequence_side(pair["right"], "right", save_seq_dir, args.no_stag))

        manifest["sequences"].append(
            {
                "seq": seq_name,
                "left_source": pair["left"].name,
                "right_source": pair["right"].name if pair["right"] else None,
                "side_delta_seconds": pair["side_delta_seconds"],
                "status": pair["status"],
                "devices": devices,
            }
        )

    if args.copy_calibration:
        manifest["copied_calibration"] = copy_calibration_files(
            args.subject, Path(args.calibration_root), output_seq_dirs
        )

    manifest_dir = raw_root / "manifests" / args.subject
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "process_sensor_auto.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
