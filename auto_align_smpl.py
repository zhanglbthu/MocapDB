import argparse
import json
import os
from pathlib import Path
import warnings

import inspect
import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    warnings.simplefilter("ignore", DeprecationWarning)
    for _name, _value in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }.items():
        if not hasattr(np, _name):
            setattr(np, _name, _value)
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

import articulate as art
import torch

from config import paths
from utils.alignment_utils import (
    crop_tensor_pair,
    estimate_alignment_bias,
    save_alignment_report,
)


def syn_acc(v, smooth_n=2, fps=30):
    mid = smooth_n // 2
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * fps**2 for i in range(0, v.shape[0] - 2)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    if mid != 0:
        acc[smooth_n:-smooth_n] = torch.stack(
            [
                (v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * fps**2 / smooth_n**2
                for i in range(0, v.shape[0] - smooth_n * 2)
            ]
        )
    return acc


def sorted_pt_sequences(seq_dir):
    return sorted(
        [p for p in Path(seq_dir).iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )


def load_smpl_sequence(seq_dir):
    smpl_dir = Path(seq_dir) / "smpl"
    pose_files = sorted(smpl_dir.glob("smpl_pose_*.pt"))
    tran_files = sorted(smpl_dir.glob("smpl_tran_*.pt"))
    if len(pose_files) != 1 or len(tran_files) != 1:
        raise FileNotFoundError(f"Expected one pose/tran file under {smpl_dir}")
    return torch.load(pose_files[0], map_location="cpu"), torch.load(tran_files[0], map_location="cpu")


DEVICE_SIGNAL_PRESETS = {
    "zhenhong_0529": {
        "left_wrist": 0,  # Watch_left
        "head": 3,  # Headset
    }
}


DEVICE_REPORT_SIGNAL_PRESETS = {
    "zhenhong_0529": {
        "phone_right": 2,
    }
}

FULL_DEVICE_ORDER = ["Watch_left", "Watch_right", "Phone_left", "Phone_right", "Headset", "STag_left", "STag_right"]
VI_MASK = [1961, 5424, 876, 4362, 411, 3365, 6765]
JI_MASK = [18, 19, 1, 2, 15, 7, 8]
DEVICE_FILL_PRESETS = {
    "zhenhong_0529": {
        "real_to_full": {
            0: 0,  # Watch_left
            1: 2,  # Phone_left
            2: 3,  # Phone_right
            3: 4,  # Headset
        },
        "synthetic_indices": [1, 5, 6],
    }
}


def parse_signal_map(items):
    signal_map = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Invalid --signal-map item: {item}. Expected name=index.")
        name, index = item.split("=", 1)
        signal_map[name.strip()] = int(index)
    return signal_map


def build_sensor_signals(sensor_data, signal_map=None):
    aM = sensor_data["aM"].cpu()
    signals = {}
    if signal_map is None:
        signal_map = {"left_wrist": 0, "right_wrist": 1, "head": 4}
    for name, index in signal_map.items():
        if 0 <= index < aM.shape[1]:
            signals[name] = torch.linalg.norm(aM[:, index], dim=-1).numpy()
    return signals


def build_mocap_signals(body_model, smpl_pose, fps):
    shape = torch.zeros(size=[10])
    tran = torch.zeros(size=[len(smpl_pose), 3])
    _, joint = body_model.forward_kinematics(smpl_pose, shape, tran, calc_mesh=False)
    signals = {}
    joint_map = {"left_wrist": 18, "right_wrist": 19, "head": 15, "phone_left": 1, "phone_right": 2}
    for name, joint_idx in joint_map.items():
        acc = syn_acc(joint[:, [joint_idx]], fps=fps)
        signals[name] = torch.linalg.norm(acc, dim=-1).reshape(-1).numpy()
    return signals


def synthesize_device_imu(body_model, smpl_pose, smpl_tran, synthetic_indices, fps=30):
    shape = torch.zeros(size=[10])
    pose_global, _, vertex = body_model.forward_kinematics(smpl_pose, shape, smpl_tran, calc_mesh=True)
    synth_aM = {}
    synth_RMB = {}
    for full_idx in synthetic_indices:
        vertex_idx = VI_MASK[full_idx]
        joint_idx = JI_MASK[full_idx]
        synth_aM[full_idx] = syn_acc(vertex[:, [vertex_idx]], fps=fps).reshape(len(smpl_pose), 3)
        synth_RMB[full_idx] = pose_global[:, joint_idx].float()
    return synth_aM, synth_RMB


def complete_aM_RMB_to_full_devices(output, body_model, fill_config, fps=30):
    if not fill_config:
        return
    real_aM = output["aM"]
    real_RMB = output["RMB"]
    n = real_aM.shape[0]
    full_aM = torch.zeros(n, len(FULL_DEVICE_ORDER), 3, dtype=real_aM.dtype)
    full_RMB = torch.eye(3, dtype=real_RMB.dtype).repeat(n, len(FULL_DEVICE_ORDER), 1, 1)

    for src_idx, dst_idx in fill_config["real_to_full"].items():
        if src_idx < real_aM.shape[1]:
            full_aM[:, dst_idx] = real_aM[:, src_idx]
            full_RMB[:, dst_idx] = real_RMB[:, src_idx]

    synth_aM, synth_RMB = synthesize_device_imu(
        body_model,
        output["pose_gt"],
        output["tran_gt"],
        fill_config["synthetic_indices"],
        fps=fps,
    )
    for dst_idx, value in synth_aM.items():
        full_aM[:, dst_idx] = value.to(dtype=real_aM.dtype)
        full_RMB[:, dst_idx] = synth_RMB[dst_idx].to(dtype=real_RMB.dtype)

    output["aM_original"] = real_aM.clone()
    output["RMB_original"] = real_RMB.clone()
    output["aM"] = full_aM
    output["RMB"] = full_RMB
    output["device_order"] = FULL_DEVICE_ORDER
    output["synthetic_device_indices"] = sorted(fill_config["synthetic_indices"])
    output["vi_mask"] = VI_MASK
    output["ji_mask"] = JI_MASK


def align_and_save(sensor_data, smpl_pose, smpl_tran, bias, output_path, body_model=None, fill_config=None, fps=30):
    output = {}
    keys_to_align = ["aM", "RMB", "acc", "gyro", "mag", "quaternion", "linear_acc", "ppg"]
    sensor_ref = sensor_data["aM"]
    _, pose_aligned = crop_tensor_pair(sensor_ref, smpl_pose, bias)
    _, tran_aligned = crop_tensor_pair(sensor_ref, smpl_tran, bias)
    n = len(pose_aligned)

    for key, value in sensor_data.items():
        if key in keys_to_align:
            aligned, _ = crop_tensor_pair(value, smpl_pose, bias)
            output[key] = aligned[:n].clone()
        else:
            output[key] = value.clone() if hasattr(value, "clone") else value

    output["pose_gt"] = pose_aligned[:n].clone()
    output["tran_gt"] = tran_aligned[:n].clone()
    output["frame_bias"] = int(bias)
    if body_model is not None and fill_config is not None:
        complete_aM_RMB_to_full_devices(output, body_model, fill_config, fps=fps)
    torch.save(output, output_path)
    return n


def load_manual_bias(processed_dir, seq_idx):
    path = Path(processed_dir) / f"{seq_idx}.pt"
    if not path.exists():
        return None
    try:
        data = torch.load(path, map_location="cpu")
    except Exception:
        return None
    return int(data["frame_bias"]) if isinstance(data, dict) and "frame_bias" in data else None


def main():
    parser = argparse.ArgumentParser(description="Automatically align MocapDB sensor_data.pt with SMPL sequences.")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="data/auto_processed")
    parser.add_argument("--report-dir", default="data/alignment_reports")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-shift-sec", type=float, default=120.0)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--min-score", type=float, default=0.53)
    parser.add_argument("--batch-bias-window", type=int, default=220)
    parser.add_argument("--disable-batch-prior", action="store_true")
    parser.add_argument("--save-review", action="store_true", help="Also save review_recommended sequences.")
    parser.add_argument("--save-all", action="store_true", help="Save every sequence with an estimated bias.")
    parser.add_argument(
        "--signal-map",
        nargs="*",
        help="Override sensor channel mapping, e.g. left_wrist=0 head=3.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    sensor_dir = data_dir / "raw" / "sensor" / args.subject
    smpl_dir = data_dir / "raw" / "smpl" / args.subject
    processed_dir = data_dir / "processed" / args.subject
    output_dir = Path(args.output_dir) / args.subject
    report_dir = Path(args.report_dir) / args.subject
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    sensor_seqs = sorted([p for p in sensor_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    smpl_seqs = sorted_pt_sequences(smpl_dir)
    if len(sensor_seqs) != len(smpl_seqs):
        raise RuntimeError(f"Sequence count mismatch: sensor={len(sensor_seqs)}, smpl={len(smpl_seqs)}")

    body_model = art.ParametricModel(paths.smpl_file)
    signal_map = parse_signal_map(args.signal_map) or DEVICE_SIGNAL_PRESETS.get(args.subject)
    report_signal_map = {}
    if signal_map:
        report_signal_map.update(signal_map)
    report_signal_map.update(DEVICE_REPORT_SIGNAL_PRESETS.get(args.subject, {}))
    fill_config = DEVICE_FILL_PRESETS.get(args.subject)
    manifest = []
    contexts = []

    for idx, (sensor_seq, smpl_seq) in enumerate(zip(sensor_seqs, smpl_seqs), start=1):
        seq_name = f"{idx}"
        print(f"Processing {args.subject} seq {idx}: {sensor_seq.name} <-> {smpl_seq.name}")
        sensor_path = sensor_seq / "sensor_data.pt"
        if not sensor_path.exists():
            manifest.append({"seq": idx, "status": "missing_sensor", "saved": False})
            continue

        sensor_data = torch.load(sensor_path, map_location="cpu")
        smpl_pose, smpl_tran = load_smpl_sequence(smpl_seq)

        sensor_signals = build_sensor_signals(sensor_data, signal_map=signal_map)
        if not sensor_signals:
            raise RuntimeError(f"No usable sensor alignment signals for {sensor_seq}")
        mocap_signals = build_mocap_signals(body_model, smpl_pose, fps=args.fps)
        result = estimate_alignment_bias(
            sensor_signals=sensor_signals,
            mocap_signals=mocap_signals,
            fps=args.fps,
            max_shift_sec=args.max_shift_sec,
        )

        report_sensor_signals = build_sensor_signals(sensor_data, signal_map=report_signal_map)
        save_alignment_report(report_dir, seq_name, result, report_sensor_signals, mocap_signals, fps=args.fps)

        manual_bias = load_manual_bias(processed_dir, idx)

        item = {
            "seq": idx,
            "sensor_seq": sensor_seq.name,
            "smpl_seq": smpl_seq.name,
            "signal_map": signal_map,
            "auto_bias": result.best_bias,
            "score": result.score,
            "confidence": result.confidence,
            "status": result.status,
            "saved": False,
            "manual_bias": manual_bias,
            "bias_error_vs_manual": None if manual_bias is None else abs(result.best_bias - manual_bias),
            "output": None,
        }
        print(
            f"  estimated bias={result.best_bias} confidence={result.confidence:.3f} "
            f"score={result.score:.3f} status={result.status}"
        )

        with open(report_dir / f"{seq_name}_alignment.json", "w", encoding="utf-8") as f:
            json.dump({**item, "result": result.to_dict()}, f, indent=2)
        manifest.append(item)
        contexts.append(
            {
                "idx": idx,
                "sensor_data": sensor_data,
                "smpl_pose": smpl_pose,
                "smpl_tran": smpl_tran,
                "result": result,
            }
        )

    bias_pool = [
        ctx["result"].best_bias
        for ctx in contexts
        if ctx["result"].score >= args.min_score and ctx["result"].best_bias >= 0
    ]
    median_bias = int(round(float(np.median(bias_pool)))) if bias_pool else None

    for item, ctx in zip(manifest, contexts):
        result = ctx["result"]
        batch_inlier = True
        if not args.disable_batch_prior and median_bias is not None:
            batch_inlier = abs(result.best_bias - median_bias) <= args.batch_bias_window
        item["batch_median_bias"] = median_bias
        item["batch_inlier"] = bool(batch_inlier)

        should_save = args.save_all or (
            result.score >= args.min_score
            and result.confidence >= args.min_confidence
            and result.best_bias >= 0
            and batch_inlier
        )
        if not should_save:
            if not batch_inlier:
                item["status"] = "batch_outlier"
            elif result.score < args.min_score:
                item["status"] = "low_score"
            print(
                f"  skipped seq={item['seq']} bias={result.best_bias} score={result.score:.3f} "
                f"batch_inlier={batch_inlier}"
            )
            continue

        output_path = output_dir / f"{item['seq']}.pt"
        frames = align_and_save(
            ctx["sensor_data"],
            ctx["smpl_pose"],
            ctx["smpl_tran"],
            result.best_bias,
            output_path,
            body_model=body_model,
            fill_config=fill_config,
            fps=args.fps,
        )
        item["saved"] = True
        item["status"] = "saved_all" if args.save_all else "auto_accept"
        item["output"] = str(output_path)
        item["frames"] = frames
        print(
            f"  saved seq={item['seq']} bias={result.best_bias} confidence={result.confidence:.3f} "
            f"score={result.score:.3f} frames={frames}"
        )

        with open(report_dir / f"{item['seq']}_alignment.json", "r", encoding="utf-8") as f:
            detail = json.load(f)
        detail.update(item)
        with open(report_dir / f"{item['seq']}_alignment.json", "w", encoding="utf-8") as f:
            json.dump(detail, f, indent=2)

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(report_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    saved = sum(1 for item in manifest if item["saved"])
    print(f"Done. saved={saved}/{len(manifest)} output={output_dir} reports={report_dir}")


if __name__ == "__main__":
    main()
