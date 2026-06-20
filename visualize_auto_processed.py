import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

import articulate as art
from config import amass, paths
from utils.model_utils import load_model


def natural_pt_key(path):
    try:
        return int(path.stem)
    except ValueError:
        return path.stem


def run_inference(data, net, device):
    aM = data["aM"].float() / amass.acc_scale
    RMB = data["RMB"].float()
    model_input = torch.cat((aM[:, :5].flatten(1), RMB[:, :5].flatten(1)), dim=1).to(device)

    net.reset()
    poses = []
    with torch.no_grad():
        for frame in range(model_input.shape[0]):
            pose = net.forward_frame(model_input[frame]).view(24, 3, 3)
            poses.append(pose.detach().cpu())
    return torch.stack(poses)


def joints_from_pose(body_model, pose, chunk_size=1024):
    joints = []
    shape = torch.zeros(10)
    for start in range(0, len(pose), chunk_size):
        chunk = pose[start : start + chunk_size].float()
        _, joint = body_model.forward_kinematics(chunk, shape, None, calc_mesh=False)
        joint = joint - joint[:, :1]
        joints.append(joint.cpu())
    return torch.cat(joints, dim=0).numpy()


def project_points(points, center_x, center_y, scale):
    xy = points[:, [0, 1]].copy()
    xy[:, 1] *= -1.0
    out = xy * scale
    out[:, 0] += center_x
    out[:, 1] += center_y
    return out.astype(np.int32)


def draw_skeleton(frame, joints, parents, center_x, center_y, scale, color, label):
    pts = project_points(joints, center_x, center_y, scale)
    for joint_idx, parent_idx in enumerate(parents):
        if parent_idx is None or parent_idx < 0:
            continue
        p0 = tuple(pts[parent_idx])
        p1 = tuple(pts[joint_idx])
        cv2.line(frame, p0, p1, color, 3, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, tuple(p), 4, color, -1, cv2.LINE_AA)
    cv2.putText(frame, label, (center_x - 90, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)


def render_video(pred_joints, gt_joints, parents, output_path, seq_name, frame_bias, fps=30, size=(1280, 720)):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    all_xy = np.concatenate([pred_joints[:, :, [0, 1]], gt_joints[:, :, [0, 1]]], axis=1)
    span = np.percentile(all_xy.reshape(-1, 2), 98, axis=0) - np.percentile(all_xy.reshape(-1, 2), 2, axis=0)
    scale = float(min(width * 0.34 / max(span[0], 1e-3), height * 0.72 / max(span[1], 1e-3)))
    scale = float(np.clip(scale, 160.0, 420.0))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    n = min(len(pred_joints), len(gt_joints))
    left_center = (width // 4, int(height * 0.62))
    right_center = (width * 3 // 4, int(height * 0.62))
    for frame_idx in range(n):
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        cv2.line(canvas, (width // 2, 0), (width // 2, height), (220, 220, 220), 1)
        draw_skeleton(
            canvas,
            pred_joints[frame_idx],
            parents,
            left_center[0],
            left_center[1],
            scale,
            (40, 105, 220),
            "pred_pose",
        )
        draw_skeleton(
            canvas,
            gt_joints[frame_idx],
            parents,
            right_center[0],
            right_center[1],
            scale,
            (40, 155, 80),
            "pose_gt",
        )
        cv2.putText(
            canvas,
            f"{seq_name}  frame {frame_idx + 1}/{n}  frame_bias={frame_bias}",
            (36, height - 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )
        writer.write(canvas)
    writer.release()


def process_sequence(pt_path, output_dir, net, body_model, device, fps, overwrite_videos, overwrite_pred):
    data = torch.load(pt_path, map_location="cpu")
    if overwrite_pred or "pose_pred" not in data:
        data["pose_pred"] = run_inference(data, net, device)
        torch.save(data, pt_path)

    output_path = output_dir / f"{pt_path.stem}.mp4"
    if output_path.exists() and not overwrite_videos:
        return "exists"

    pred_pose = data["pose_pred"].view(-1, 24, 3, 3).cpu()
    gt_pose = data["pose_gt"].view(-1, 24, 3, 3).cpu()
    n = min(len(pred_pose), len(gt_pose))
    pred_joints = joints_from_pose(body_model, pred_pose[:n])
    gt_joints = joints_from_pose(body_model, gt_pose[:n])
    render_video(
        pred_joints,
        gt_joints,
        body_model.parent,
        output_path,
        pt_path.stem,
        data.get("frame_bias", "NA"),
        fps=fps,
    )
    return "rendered"


def main():
    parser = argparse.ArgumentParser(description="Infer pose_pred and render pred/gt side-by-side videos.")
    parser.add_argument("--input-dir", default="data/auto_processed/zhenhong_0529")
    parser.add_argument("--output-dir", default="data/auto_processed/zhenhong_0529_videos")
    parser.add_argument("--ckpt", default="data/ckpt/full_model.pth")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--overwrite-videos", action="store_true")
    parser.add_argument("--overwrite-pred", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    pt_files = sorted(input_dir.glob("*.pt"), key=natural_pt_key)
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {input_dir}")

    net = load_model(args.ckpt)
    net.eval()
    device = next(net.parameters()).device
    body_model = art.ParametricModel(paths.smpl_file)

    counts = {"rendered": 0, "exists": 0}
    for pt_path in tqdm(pt_files, desc="visualizing"):
        status = process_sequence(
            pt_path,
            output_dir,
            net,
            body_model,
            device,
            args.fps,
            args.overwrite_videos,
            args.overwrite_pred,
        )
        counts[status] = counts.get(status, 0) + 1
    print(f"Done: {counts}, output={output_dir}")


if __name__ == "__main__":
    main()
