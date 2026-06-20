import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import torch


DEFAULT_CALIBRATION_ROOT = r"E:\Research\project\daily_mocap\dataset\code\SmartWear\data\recording"
DEFAULT_SENSOR_SOURCE_ROOT = r"E:\Research\project\daily_mocap\dataset\data\sensor"
DEFAULT_RAW_ROOT = "data/raw"


def run(cmd, cwd):
    print("\n$ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=cwd, check=True)


def count_dirs(path):
    path = Path(path)
    return len([p for p in path.iterdir() if p.is_dir()]) if path.exists() else 0


def count_files(path, pattern="*"):
    path = Path(path)
    return len(list(path.glob(pattern))) if path.exists() else 0


def infer_date(subject, raw_root, calibration_root):
    candidates = []
    for root in [Path(raw_root) / "calibration" / subject, Path(calibration_root) / subject]:
        if not root.exists():
            continue
        for path in root.glob("*.pt"):
            match = re.search(r"_(\d{8})_", path.name)
            if match:
                candidates.append(match.group(1))
    if candidates:
        return sorted(set(candidates))[0]

    suffix = subject.rsplit("_", 1)[-1]
    if re.fullmatch(r"\d{4}", suffix):
        return f"2026{suffix}"
    return None


def powershell_exe():
    candidates = [
        Path(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"),
        Path(r"C:\Windows\SysWOW64\WindowsPowerShell\v1.0\powershell.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return "powershell"


def raw_available(raw_root, subject):
    raw_root = Path(raw_root)
    return (
        count_dirs(raw_root / "sensor_raw" / subject / "left") > 0
        and count_dirs(raw_root / "sensor_raw" / subject / "right") > 0
        and count_files(raw_root / "calibration" / subject, "*.pt") > 0
    )


def smpl_available(raw_root, subject):
    return count_dirs(Path(raw_root) / "smpl" / subject) > 0


def infer_no_stag(subject, raw_root, calibration_root):
    candidates = []
    for root in [Path(raw_root) / "calibration" / subject, Path(calibration_root) / subject]:
        if root.exists():
            candidates.extend(sorted(root.glob("*.pt")))
    for path in candidates:
        try:
            data = torch.load(path, map_location="cpu")
            if "RMI" in data:
                return int(data["RMI"].shape[0]) < 7
        except Exception:
            continue
    return False


def extract_raw(args, repo_root, date):
    left_root = Path(args.left_sensor_root) if args.left_sensor_root else Path(args.sensor_source_root) / date / "left"
    right_root = Path(args.right_sensor_root) if args.right_sensor_root else Path(args.sensor_source_root) / date / "right"
    script = repo_root / "tools" / "extract_raw_records.ps1"
    ps = powershell_exe()

    for side, sensor_root in [("left", left_root), ("right", right_root)]:
        if not sensor_root.exists():
            raise FileNotFoundError(f"Sensor source for {side} not found: {sensor_root}")
        cmd = [
            ps,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            script,
            "-Subject",
            args.subject,
            "-Side",
            side,
            "-Date",
            date,
            "-SensorRoot",
            sensor_root,
            "-CalibrationRoot",
            args.calibration_root,
            "-OutputRawRoot",
            Path(args.raw_root).resolve(),
            "-MinFolderSizeMB",
            str(args.min_folder_size_mb),
            "-MaxDeltaMinutes",
            str(args.max_delta_minutes),
            "-Overwrite",
        ]
        if args.no_stag:
            cmd.append("-NoSTag")
        run(cmd, repo_root)


def run_pipeline(args):
    repo_root = Path(__file__).resolve().parent
    raw_root = Path(args.raw_root)
    date = args.date or infer_date(args.subject, raw_root, args.calibration_root)
    if not date:
        raise ValueError("Cannot infer date. Provide --date YYYYMMDD.")

    print(f"Subject: {args.subject}")
    print(f"Date:    {date}")
    args.no_stag = bool(args.no_stag or infer_no_stag(args.subject, raw_root, args.calibration_root))
    print(f"No STag: {args.no_stag}")

    if raw_available(raw_root, args.subject):
        print("Raw sensor/calibration already exists under data/raw; using existing raw files.")
    else:
        print("Raw files are incomplete under data/raw; extracting from source directories.")
        extract_raw(args, repo_root, date)

    if not smpl_available(raw_root, args.subject):
        raise FileNotFoundError(f"SMPL directory missing: {raw_root / 'smpl' / args.subject}")

    if not args.skip_process_sensor:
        process_cmd = [
            sys.executable,
            "process_sensor_auto.py",
            "--subject",
            args.subject,
            "--raw-root",
            args.raw_root,
            "--copy-calibration",
            "--overwrite",
        ]
        if args.no_stag:
            process_cmd.append("--no-stag")
        run(process_cmd, repo_root)

    if not args.skip_align_sensor:
        align_cmd = [
            sys.executable,
            "align_sensor_auto.py",
            "--subject",
            args.subject,
            "--sensor-root",
            str(raw_root / "sensor"),
            "--calibration-root",
            str(raw_root / "calibration"),
            "--overwrite",
            "--peak-mode",
            args.peak_mode,
            "--plot-offset",
            str(args.plot_offset),
        ]
        if args.device_order:
            align_cmd.extend(["--device-order", *args.device_order])
        run(align_cmd, repo_root)

    if not args.skip_align_smpl:
        smpl_cmd = [
            sys.executable,
            "auto_align_smpl.py",
            "--subject",
            args.subject,
            "--data-dir",
            str(raw_root.parent if raw_root.name == "raw" else "data"),
            "--output-dir",
            args.output_dir,
            "--report-dir",
            args.report_dir,
        ]
        if args.save_all:
            smpl_cmd.append("--save-all")
        if args.signal_map:
            smpl_cmd.extend(["--signal-map", *args.signal_map])
        run(smpl_cmd, repo_root)

    if not args.skip_visualize:
        vis_cmd = [
            sys.executable,
            "visualize_auto_processed.py",
            "--input-dir",
            str(Path(args.output_dir) / args.subject),
            "--output-dir",
            str(Path(args.output_dir) / f"{args.subject}_videos"),
            "--ckpt",
            args.ckpt,
            "--overwrite-videos",
            "--overwrite-pred",
        ]
        run(vis_cmd, repo_root)

    summary = {
        "subject": args.subject,
        "date": date,
        "raw_left": count_dirs(raw_root / "sensor_raw" / args.subject / "left"),
        "raw_right": count_dirs(raw_root / "sensor_raw" / args.subject / "right"),
        "sensor_seq": count_dirs(raw_root / "sensor" / args.subject),
        "processed_pt": count_files(Path(args.output_dir) / args.subject, "*.pt"),
        "videos": count_files(Path(args.output_dir) / f"{args.subject}_videos", "*.mp4"),
    }
    manifest_dir = raw_root / "manifests" / args.subject
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "full_pipeline.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nPipeline summary:")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Run the full MocapDB subject processing pipeline.")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--date", help="Recording date, e.g. 20260529. Inferred from calibration when omitted.")
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--calibration-root", default=DEFAULT_CALIBRATION_ROOT)
    parser.add_argument("--sensor-source-root", default=DEFAULT_SENSOR_SOURCE_ROOT)
    parser.add_argument("--left-sensor-root")
    parser.add_argument("--right-sensor-root")
    parser.add_argument("--min-folder-size-mb", type=int, default=3)
    parser.add_argument("--max-delta-minutes", type=int, default=10)
    parser.add_argument("--no-stag", action="store_true", help="Use for subjects without STag devices.")
    parser.add_argument("--device-order", nargs="+")
    parser.add_argument("--signal-map", nargs="*")
    parser.add_argument("--peak-mode", choices=["individual_max", "consensus"], default="individual_max")
    parser.add_argument("--plot-offset", type=float, default=60.0)
    parser.add_argument("--output-dir", default="data/auto_processed")
    parser.add_argument("--report-dir", default="data/alignment_reports")
    parser.add_argument("--ckpt", default="data/ckpt/full_model.pth")
    parser.add_argument("--save-all", action="store_true", default=True)
    parser.add_argument("--skip-process-sensor", action="store_true")
    parser.add_argument("--skip-align-sensor", action="store_true")
    parser.add_argument("--skip-align-smpl", action="store_true")
    parser.add_argument("--skip-visualize", action="store_true")
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
