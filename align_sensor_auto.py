import argparse
import json
import re
from pathlib import Path

import articulate as art
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from align_sensor import downsample, find_csv_by_keywords


DEVICE_ORDER_PRESETS = {
    "zhenhong_0529": ["Watch_left", "Phone_left", "Phone_right", "Headset"],
    "chaoran_0529": ["Watch_left", "Watch_right", "Phone_left", "Phone_right", "Headset"],
}
FULL_DEVICE_ORDER = ["Watch_left", "Watch_right", "Phone_left", "Phone_right", "Headset", "STag_left", "STag_right"]
FIVE_DEVICE_ORDER = ["Watch_left", "Watch_right", "Phone_left", "Phone_right", "Headset"]


KEYS_IN_DEVICE = {
    "Headset": ["acc_headsetL", "gyro_headsetL", "quaternion_left"],
    "Phone": ["acc", "gyro", "linear_acc", "magnetic", "rotation"],
    "Watch": ["acc", "gyro", "line_acc", "mag", "ppg", "quaternion"],
    "STag": ["acc", "gyro", "quaternion"],
}


def calibration_sort_key(path):
    match = re.match(r"^(\d+)(?:_(\d+))?_(\d{8})_(\d{6})\.pt$", path.name)
    if not match:
        return (999999, 999999, path.name)
    part = int(match.group(2)) if match.group(2) else 0
    return (int(match.group(1)), part, path.name)


def list_calibration_files(calibration_dir):
    raw_files = sorted(calibration_dir.glob("*.pt"), key=calibration_sort_key)
    parsed = []
    for path in raw_files:
        match = re.match(r"^(\d+)(?:_(\d+))?_(\d{8})_(\d{6})\.pt$", path.name)
        parsed.append((path, match))
    has_repeated_trials = any(match and match.group(2) for _, match in parsed)
    if has_repeated_trials:
        return raw_files

    grouped = {}
    for path, match in parsed:
        if not match:
            grouped[path.name] = path
            continue
        seq = int(match.group(1))
        prev = grouped.get(seq)
        if prev is None or path.stat().st_size > prev.stat().st_size:
            grouped[seq] = path
    return [grouped[key] for key in sorted(grouped, key=lambda x: (999999, x) if isinstance(x, str) else (x, ""))]


def infer_device_order(subject, sensor_subject, calibration_files):
    preset = DEVICE_ORDER_PRESETS.get(subject)
    if preset:
        return preset
    seq_dirs = sorted([p for p in sensor_subject.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not seq_dirs:
        return None
    present = {p.name for p in seq_dirs[0].iterdir() if p.is_dir()}
    if all(device in present for device in FULL_DEVICE_ORDER):
        return FULL_DEVICE_ORDER
    if all(device in present for device in FIVE_DEVICE_ORDER):
        return FIVE_DEVICE_ORDER
    if calibration_files:
        try:
            cali = torch.load(calibration_files[0], map_location="cpu")
            count = int(cali["RMI"].shape[0])
            if count == len(FULL_DEVICE_ORDER):
                return [d for d in FULL_DEVICE_ORDER if d in present]
            if count == len(FIVE_DEVICE_ORDER):
                return [d for d in FIVE_DEVICE_ORDER if d in present]
        except Exception:
            pass
    return sorted(present)


def infer_device_type(device_name):
    if device_name == "Headset":
        return "Headset"
    if device_name.startswith("Phone"):
        return "Phone"
    if device_name.startswith("Watch"):
        return "Watch"
    if device_name.startswith("STag"):
        return "STag"
    raise ValueError(f"Unsupported device: {device_name}")


def read_csv_tail(device_dir, key, start_index, length):
    csv_file = find_csv_by_keywords(str(device_dir), [key])
    if csv_file is None:
        raise FileNotFoundError(f"Missing {key} in {device_dir}")
    df = pd.read_csv(csv_file)
    data = df.iloc[start_index : start_index + length].reset_index(drop=True)
    return data.iloc[:, -11:].to_numpy() if key == "ppg" else data.iloc[:, -4:].to_numpy() if key in ["rotation", "quaternion", "quaternion_left"] else data.iloc[:, -3:].to_numpy()


def get_acc_norm_from_df(df):
    acc_cols = [col for col in df.columns if "acc" in col.lower() and "time" not in col.lower()]
    acc_values = df[acc_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    return np.linalg.norm(acc_values, axis=1) - 9.8


def smooth_signal(x, window=5):
    if window <= 1 or len(x) < window:
        return x
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(x, kernel, mode="same")


def find_peak_candidates(acc_norm, search_frames, top_k=8, min_distance=30):
    signal = smooth_signal(acc_norm[:search_frames], window=5)
    if len(signal) < 3:
        return [{"frame": int(np.argmax(signal)), "value": float(np.max(signal))}]

    local = np.where((signal[1:-1] > signal[:-2]) & (signal[1:-1] >= signal[2:]))[0] + 1
    if local.size == 0:
        local = np.array([int(np.argmax(signal))])
    order = local[np.argsort(signal[local])[::-1]]
    selected = []
    for frame in order:
        frame = int(frame)
        if all(abs(frame - item["frame"]) >= min_distance for item in selected):
            selected.append({"frame": frame, "value": float(signal[frame])})
        if len(selected) >= top_k:
            break
    return selected


def select_consensus_peaks(seq_dir, device_order, search_frames, top_k, consensus_tolerance):
    candidates_by_device = {}
    for device_name in device_order:
        acc_file = find_csv_by_keywords(str(seq_dir / device_name), ["acc"])
        if acc_file is None:
            raise FileNotFoundError(f"Missing acc csv in {seq_dir / device_name}")
        acc_df = pd.read_csv(acc_file)
        acc_norm = get_acc_norm_from_df(acc_df)
        candidates_by_device[device_name] = find_peak_candidates(acc_norm, search_frames, top_k=top_k)

    consensus_frames = sorted(
        {candidate["frame"] for candidates in candidates_by_device.values() for candidate in candidates}
    )
    best = None
    for consensus in consensus_frames:
        chosen = {}
        total_distance = 0.0
        total_strength = 0.0
        max_distance = 0.0
        for device_name, candidates in candidates_by_device.items():
            nearest = min(candidates, key=lambda item: abs(item["frame"] - consensus))
            distance = abs(nearest["frame"] - consensus)
            chosen[device_name] = nearest
            total_distance += distance
            total_strength += nearest["value"]
            max_distance = max(max_distance, distance)
        # Distance dominates; strength only breaks ties among similarly consistent groups.
        score = total_distance + 0.5 * max_distance - 0.01 * total_strength
        if best is None or score < best["score"]:
            best = {
                "score": float(score),
                "consensus_frame": int(consensus),
                "chosen": chosen,
                "total_distance": float(total_distance),
                "max_distance": float(max_distance),
            }

    status = "consensus"
    if best["max_distance"] > consensus_tolerance:
        status = "low_consensus"
    return best, candidates_by_device, status


def collect_device_data(seq_dir, device_order, jump_search_frames, pre_peak_frames, peak_top_k, consensus_tolerance):
    starts = {}
    per_device = {}
    min_len = None
    consensus, candidates_by_device, consensus_status = select_consensus_peaks(
        seq_dir,
        device_order,
        search_frames=jump_search_frames,
        top_k=peak_top_k,
        consensus_tolerance=consensus_tolerance,
    )
    for device_name in device_order:
        device_dir = seq_dir / device_name
        device_type = infer_device_type(device_name)
        peak = int(consensus["chosen"][device_name]["frame"])
        start = max(0, peak - pre_peak_frames)
        starts[device_name] = {
            "peak": peak,
            "start": int(start),
            "peak_in_aligned": int(peak - start),
            "peak_value": float(consensus["chosen"][device_name]["value"]),
        }

        device_data = {}
        for key in KEYS_IN_DEVICE[device_type]:
            arr = read_csv_tail(device_dir, key, start, 10**12)
            device_data[key] = arr
            min_len = arr.shape[0] if min_len is None else min(min_len, arr.shape[0])
        per_device[device_name] = device_data

    for device_name in per_device:
        for key in per_device[device_name]:
            per_device[device_name][key] = per_device[device_name][key][:min_len]
    diagnostics = {
        "status": consensus_status,
        "consensus_frame": consensus["consensus_frame"],
        "total_distance": consensus["total_distance"],
        "max_distance": consensus["max_distance"],
        "candidates": candidates_by_device,
    }
    return starts, per_device, min_len, diagnostics


def collect_device_data_individual_max(seq_dir, device_order, jump_search_frames, pre_peak_frames):
    starts = {}
    per_device = {}
    min_len = None
    candidates_by_device = {}
    for device_name in device_order:
        device_dir = seq_dir / device_name
        device_type = infer_device_type(device_name)
        acc_file = find_csv_by_keywords(str(device_dir), ["acc"])
        if acc_file is None:
            raise FileNotFoundError(f"Missing acc csv in {device_dir}")
        acc_df = pd.read_csv(acc_file)
        acc_norm = get_acc_norm_from_df(acc_df)
        search = acc_norm[:jump_search_frames]
        peak = int(np.nanargmax(search))
        start = max(0, peak - pre_peak_frames)
        starts[device_name] = {
            "peak": peak,
            "start": int(start),
            "peak_in_aligned": int(peak - start),
            "peak_value": float(search[peak]),
        }
        candidates_by_device[device_name] = [{"frame": peak, "value": float(search[peak])}]

        device_data = {}
        for key in KEYS_IN_DEVICE[device_type]:
            arr = read_csv_tail(device_dir, key, start, 10**12)
            device_data[key] = arr
            min_len = arr.shape[0] if min_len is None else min(min_len, arr.shape[0])
        per_device[device_name] = device_data

    for device_name in per_device:
        for key in per_device[device_name]:
            per_device[device_name][key] = per_device[device_name][key][:min_len]

    peak_values = [item["peak"] for item in starts.values()]
    diagnostics = {
        "status": "individual_max",
        "max_distance": int(max(peak_values) - min(peak_values)),
        "candidates": candidates_by_device,
    }
    return starts, per_device, min_len, diagnostics


def read_acc_norm(device_dir, max_frames=None):
    acc_file = find_csv_by_keywords(str(device_dir), ["acc"])
    if acc_file is None:
        return None
    df = pd.read_csv(acc_file)
    acc_cols = [col for col in df.columns if "acc" in col.lower() and "time" not in col.lower()]
    acc_values = df[acc_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    norm = np.linalg.norm(acc_values, axis=1) - 9.8
    if max_frames is not None:
        norm = norm[:max_frames]
    return norm


def save_alignment_plot(seq_dir, device_order, starts, seq_name, max_plot_frames=None, offset=60.0):
    plt.figure(figsize=(10, 6))
    for i, device_name in enumerate(device_order):
        signal = read_acc_norm(seq_dir / device_name)
        start = starts[device_name]["start"]
        peak = starts[device_name]["peak"]
        if signal is None:
            aligned = np.array([])
        elif max_plot_frames is None:
            aligned = signal[start:]
        else:
            aligned = signal[start : start + max_plot_frames]
        peak_in_aligned = starts[device_name]["peak_in_aligned"]
        plt.plot(aligned + offset * i, linewidth=1.0, label=f"{device_name} peak={peak} start={start}")
        plt.axvline(peak_in_aligned, color="tab:red", linestyle="--", linewidth=0.7, alpha=0.28)

    plt.title(f"Acceleration Magnitude for {seq_name}")
    plt.xlabel("Sample Index")
    plt.ylabel("Acceleration Magnitude + offset")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(seq_dir / "acc_alignment.png", dpi=140)
    plt.close()


def stack_modalities(per_device, device_order):
    accs, gyros, quats = [], [], []
    mags, linear_accs, ppgs = [], [], []
    for device_name in device_order:
        device_type = infer_device_type(device_name)
        data = per_device[device_name]
        acc_key = "acc_headsetL" if device_type == "Headset" else "acc"
        gyro_key = "gyro_headsetL" if device_type == "Headset" else "gyro"
        quat_key = "quaternion_left" if device_type == "Headset" else "rotation" if device_type == "Phone" else "quaternion"
        accs.append(data[acc_key])
        gyros.append(data[gyro_key])
        quats.append(data[quat_key])
        if device_type in ["Phone", "Watch"]:
            mag_key = "magnetic" if device_type == "Phone" else "mag"
            linear_key = "linear_acc" if device_type == "Phone" else "line_acc"
            mags.append(data[mag_key])
            linear_accs.append(data[linear_key])
        if device_type == "Watch":
            ppgs.append(data["ppg"])

    tensors = {
        "acc": torch.from_numpy(np.stack(accs, axis=1)).float(),
        "gyro": torch.from_numpy(np.stack(gyros, axis=1)).float(),
        "quaternion": torch.from_numpy(np.stack(quats, axis=1)).float(),
    }
    if mags:
        tensors["mag"] = torch.from_numpy(np.stack(mags, axis=1)).float()
    if linear_accs:
        tensors["linear_acc"] = torch.from_numpy(np.stack(linear_accs, axis=1)).float()
    if ppgs:
        tensors["ppg"] = torch.from_numpy(np.stack(ppgs, axis=1)).float()
    return tensors


def apply_calibration(tensors, cali_path, device_order):
    cali = torch.load(cali_path, map_location="cpu")
    RMI, RSB = cali["RMI"].float(), cali["RSB"].float()
    if RMI.shape[0] != len(device_order):
        raise ValueError(
            f"Calibration device count {RMI.shape[0]} does not match device_order {len(device_order)}: {device_order}"
        )
    aS = tensors["acc"].clone()
    quat_IS = tensors["quaternion"].clone()
    RIS = art.math.quaternion_to_rotation_matrix(quat_IS).view(-1, aS.shape[1], 3, 3)
    aI = RIS.matmul(aS.unsqueeze(-1)).squeeze(-1) + torch.tensor([0, 0, -9.8])
    aM = RMI.matmul(aI.unsqueeze(-1)).squeeze(-1)
    RMB = RMI.matmul(RIS).matmul(RSB)
    tensors["aM"] = aM
    tensors["RMB"] = RMB
    return tensors


def downsample_all(tensors, original_fps=100, target_fps=30):
    return {key: downsample(value, original_fps=original_fps, target_fps=target_fps) for key, value in tensors.items()}


def main():
    parser = argparse.ArgumentParser(description="Align processed sensor CSVs by jump peaks and build sensor_data.pt.")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--sensor-root", default="data/raw/sensor")
    parser.add_argument("--calibration-root", default="data/raw/calibration")
    parser.add_argument("--device-order", nargs="+")
    parser.add_argument("--jump-search-frames", type=int, default=1000)
    parser.add_argument("--pre-peak-frames", type=int, default=200)
    parser.add_argument("--peak-top-k", type=int, default=8)
    parser.add_argument("--consensus-tolerance", type=int, default=80)
    parser.add_argument("--peak-mode", choices=["individual_max", "consensus"], default="individual_max")
    parser.add_argument("--plot-offset", type=float, default=60.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sensor_subject = Path(args.sensor_root) / args.subject
    seq_dirs = sorted([p for p in sensor_subject.iterdir() if p.is_dir()], key=lambda p: p.name)
    calibration_files = list_calibration_files(Path(args.calibration_root) / args.subject)
    device_order = args.device_order or infer_device_order(args.subject, sensor_subject, calibration_files)
    if not device_order:
        raise ValueError("Provide --device-order for this subject.")
    if len(seq_dirs) != len(calibration_files):
        raise ValueError(f"seq/calibration count mismatch: {len(seq_dirs)} vs {len(calibration_files)}")

    manifest = {
        "subject": args.subject,
        "device_order": device_order,
        "peak_mode": args.peak_mode,
        "jump_search_frames": args.jump_search_frames,
        "pre_peak_frames": args.pre_peak_frames,
        "plot_offset": args.plot_offset,
        "plot_range": "from_each_device_start_to_end",
        "sequences": [],
    }
    for i, (seq_dir, cali_path) in enumerate(zip(seq_dirs, calibration_files), start=1):
        save_path = seq_dir / "sensor_data.pt"
        if save_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists, use --overwrite: {save_path}")
        print(f"Aligning {seq_dir.name} with {cali_path.name}")
        if args.peak_mode == "consensus":
            starts, per_device, min_len, diagnostics = collect_device_data(
                seq_dir,
                device_order,
                args.jump_search_frames,
                args.pre_peak_frames,
                args.peak_top_k,
                args.consensus_tolerance,
            )
        else:
            starts, per_device, min_len, diagnostics = collect_device_data_individual_max(
                seq_dir,
                device_order,
                args.jump_search_frames,
                args.pre_peak_frames,
            )
        save_alignment_plot(seq_dir, device_order, starts, seq_dir.name, offset=args.plot_offset)
        tensors = stack_modalities(per_device, device_order)
        tensors = apply_calibration(tensors, cali_path, device_order)
        tensors = downsample_all(tensors)
        torch.save(tensors, save_path)
        manifest["sequences"].append(
            {
                "seq": seq_dir.name,
                "calibration": cali_path.name,
                "starts": starts,
                "peak_diagnostics": diagnostics,
                "peak_mode": args.peak_mode,
                "pre_downsample_frames": int(min_len),
                "post_downsample_frames": int(tensors["aM"].shape[0]),
                "keys": {k: list(v.shape) for k, v in tensors.items()},
            }
        )

    manifest_path = Path(args.sensor_root).parent / "manifests" / args.subject / "align_sensor_auto.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
