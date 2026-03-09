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

class MotionViewerManager:
    def __init__(self, sub_num, overlap=True, names=None):
        self.viewer = MotionViewer(sub_num, overlap, names)
        self.viewer.connect()     

    def visualize(self, pose, tran=None):
        clock = Clock()
        sub_num = len(pose)

        for i in range(len(pose[0])):
            clock.tick(60)
            self.viewer.clear_line(render=False)
            self.viewer.clear_point(render=False)
            self.viewer.clear_terrian(render=False)

            pose_list = [pose[sub][i] for sub in range(sub_num)]
            tran_list = [tran[sub][i] for sub in range(sub_num)] if tran else [torch.zeros(3) for _ in range(sub_num)]
                
            self.viewer.update_all(pose_list, tran_list, render=False)
            self.viewer.render()
            print('\r', clock.get_fps(), end='')

    def close(self):
        self.viewer.disconnect()

def process_easymocap_data(sub_dir, seq_name):
    seq_idx = seq_name.split('_')[-1]
    seq_dir = os.path.join(sub_dir, seq_name)
    smpl_dir = os.path.join(seq_dir, 'smpl')
    
    pose_name = 'smpl_pose_' + seq_idx + '.pt'
    tran_name = 'smpl_tran_' + seq_idx + '.pt'
    
    pose_path = os.path.join(smpl_dir, pose_name)
    tran_path = os.path.join(smpl_dir, tran_name)
    
    pose = torch.load(pose_path)
    tran = torch.load(tran_path)
    
    print('pose shape:', pose.shape, 'tran shape:', tran.shape)
    return pose, tran

def vis_easymocap():
    data_dir = 'data/processed'
    sub_name = 'multisys_zhanglb'
    sub_dir = os.path.join(data_dir, sub_name)
    
    seq_names = os.listdir(sub_dir)
    seq_num = len(seq_names)
    idx_list = [i for i in range(0, seq_num)]
    print('len:', len(idx_list))
    
    for seq_name in seq_names:
        pose_list, tran_list = [], []

        pose, tran = process_easymocap_data(sub_dir, seq_name)
        pose_list.append(pose)
        tran_list.append(tran)

        name_list = ['markerless']

        viewer_manager = MotionViewerManager(len(pose_list), overlap=False, names=name_list)

        viewer_manager.visualize(pose_list)
        viewer_manager.close()

def vis_easymocap_xingying():
    data_dir = 'data/aligned_debug'
    sub_name = 'multisys_zhanglb'
    sub_dir = os.path.join(data_dir, sub_name)
    seq_names = os.listdir(sub_dir)
    seq_names = sorted(seq_names, key=lambda x: int(x.split('.')[0]))
    print('seq_names:', seq_names)
    
    for i in range(len(seq_names)):
        seq_name = seq_names[i]
        print (f'Visualizing sequence {i}/{len(seq_names)}: {seq_name}...')
        
        data = torch.load(os.path.join(sub_dir, seq_name))

        pose_xy, tran_xy = data['pose'][0], data['tran'][0]
        pose_em, tran_em = data['pose_em'][0], data['tran_em'][0]
        
        # mask ignored joints
        pose_xy[:, joint_set.ignored_vis] = torch.eye(3, device=pose_xy.device)
        pose_em[:, joint_set.ignored_vis] = torch.eye(3, device=pose_em.device)
        
        # 对齐序列长度
        min_len = min(pose_xy.shape[0], pose_em.shape[0])

        pose_xy = pose_xy[:min_len]
        tran_xy = tran_xy[:min_len]

        pose_em = pose_em[:min_len]
        tran_em = tran_em[:min_len]
        
        pose_list = []
        pose_list.append(pose_xy)
        pose_list.append(pose_em)
        name_list = ['OptiMocap_'+str(i+1), 'EasyMocap']
        
        viewer_manager = MotionViewerManager(len(pose_list), overlap=False, names=name_list)
        viewer_manager.visualize(pose_list)
        viewer_manager.close()
        
if __name__ == '__main__':
    vis_easymocap_xingying()