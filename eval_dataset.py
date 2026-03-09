from pygame.time import Clock
import articulate as art
from articulate.utils.unity import MotionViewer
from articulate.utils.pygame import StreamingDataViewer
from utils import *
from config import paths, joint_set
import torch
import os
import open3d as o3d
import numpy as np
import matplotlib

body_model = art.ParametricModel(paths.smpl_file)
vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])
colors = matplotlib.colormaps['tab10'].colors

def value2color(value):
    value = torch.clamp(value, 0, 1).cpu().numpy()
    color = np.array([1, 1 - value, 1 - value])
    return color

class PoseEvaluator:
    def __init__(self):
        self._eval_fn = art.FullMotionEvaluator(paths.smpl_file, joint_mask=torch.tensor([2, 5, 16, 20]), fps=60,)

    def eval(self, pose_p, pose_t, tran_p=None, tran_t=None):
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        
        # if tran_p and tran_t are not None:
        if tran_p is not None and tran_t is not None:
            tran_p = tran_p.clone().view(-1, 3)
            tran_t = tran_t.clone().view(-1, 3)
        else:
            # initialize tran_p and tran_t to zeros if not provided
            tran_p = torch.zeros(pose_p.shape[0], 3, device=pose_p.device)
            tran_t = torch.zeros(pose_t.shape[0], 3, device=pose_t.device)

        pose_p[:, joint_set.ignored] = torch.eye(3, device=pose_p.device)
        pose_t[:, joint_set.ignored] = torch.eye(3, device=pose_t.device)

        errs = self._eval_fn(pose_p, pose_t, tran_p=tran_p, tran_t=tran_t)

        return torch.stack([
            errs[9],           # SIP
            errs[3],           # Angular
            errs[9],           # Masked Angular
            errs[0] * 100,     # Positional
            errs[7] * 100,     # Masked Positional
            errs[1] * 100,     # Mesh
            errs[4] / 100,     # Jitter
            errs[6],           # Distance
        ])

    @staticmethod
    def print(errors):
        names = [
            'SIP Error (deg)',
            'Angular Error (deg)',
            'Masked Angular Error (deg)',
            'Positional Error (cm)',
            'Masked Positional Error (cm)',
            'Mesh Error (cm)',
            'Jitter Error (100m/s^3)',
            'Distance Error (cm)',
        ]
        for i, n in enumerate(names):
            print(f"{n}: {errors[i,0]:.2f} (+/- {errors[i,1]:.2f})")

    @staticmethod
    def print_single(errors, file=None):
        names = [
            'Angular Error (deg)',
            'Mesh Error (cm)',
        ]
        max_len = max(len(n) for n in names)
        outs = []
        for i, n in enumerate([
            'SIP Error (deg)',
            'Angular Error (deg)',
            'Masked Angular Error (deg)',
            'Positional Error (cm)',
            'Masked Positional Error (cm)',
            'Mesh Error (cm)',
            'Jitter Error (100m/s^3)',
            'Distance Error (cm)',
        ]):
            if n in names:
                outs.append(f"{n:<{max_len}}: {errors[i,0]:.2f}")
        print(" | ".join(outs), file=file)

def eval_easymocap_xingying(output_dir = "data/output"):
    data_dir = 'data/aligned_debug'
    sub_name = 'multisys_zhanglb'
    sub_dir = os.path.join(data_dir, sub_name)
    seq_names = os.listdir(sub_dir)
    seq_names = sorted(seq_names, key=lambda x: int(x.split('.')[0]))
    print('seq_names:', seq_names)
    
    evaluator = PoseEvaluator()
    errs = []
    
    for i in range(len(seq_names)):
        seq_name = seq_names[i]
        print (f'Visualizing sequence {i+1}/{len(seq_names)}: {seq_name}...')
        
        data = torch.load(os.path.join(sub_dir, seq_name))

        pose_xy, tran_xy = data['pose'][0], data['tran'][0]
        pose_em, tran_em = data['pose_em'][0], data['tran_em'][0]
        
        # mask ignored joints
        pose_xy[:, joint_set.ignored_vis] = torch.eye(3, device=pose_xy.device)
        pose_em[:, joint_set.ignored_vis] = torch.eye(3, device=pose_em.device)
        
        # 对齐序列长度
        min_len = min(pose_xy.shape[0], pose_em.shape[0])
        pose_xy, tran_xy = pose_xy[:min_len], tran_xy[:min_len]
        pose_em, tran_em = pose_em[:min_len], tran_em[:min_len]
        
        err = evaluator.eval(pose_em, pose_xy)
        errs.append(err)
    
    print('Evaluation results:')
    evaluator.print(torch.stack(errs).mean(dim=0))
    
    log_dir = os.path.join(output_dir, sub_name)
    log_path = os.path.join(log_dir, sub_name + '_evaluation.txt')
    os.makedirs(log_dir, exist_ok=True)
    with open(log_path, "w") as f:
        for i, e in enumerate(errs):
            evaluator.print_single(e, file=f)
        
if __name__ == '__main__':
    eval_easymocap_xingying()