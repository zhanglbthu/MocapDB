from scipy.interpolate import interp1d
import articulate as art
import torch
import numpy as np
import matplotlib.pyplot as plt
from smpl_light import SMPLight
from articulate.math import quaternion_to_rotation_matrix, r6d_to_rotation_matrix, rotation_matrix_to_axis_angle, \
    axis_angle_to_rotation_matrix, rotation_matrix_to_r6d
# import pandas as pd
from scipy.spatial.transform import Rotation 
from config import paths

import time
import os

def syn_acc(v, smooth_n=2, fps=30):
    r"""
    Synthesize accelerations from vertex positions.
    """
    mid = smooth_n // 2
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * fps ** 2 for i in range(0, v.shape[0] - 2)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    if mid != 0:
        acc[smooth_n:-smooth_n] = torch.stack(
            [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * fps ** 2 / smooth_n ** 2
             for i in range(0, v.shape[0] - smooth_n * 2)])
    return acc

def latency_estimate(seq_1, seq_2, threadhold=20000):
    """
    算seq2比seq1延迟了多少
    """

    seq_1 = np.array(seq_1).reshape(-1)[:threadhold]
    seq_2 = np.array(seq_2).reshape(-1)[:threadhold]

    # 计算互相关
    correlation = np.correlate(seq_1, seq_2, 'full')

    # 找到互相关的峰值
    peak_index_mean = 0
    for _ in range(2):
        peak_index = np.argmax(correlation)
        # print(peak_index)
        correlation[peak_index] = 0
        peak_index_mean += peak_index
    peak_index_mean = peak_index_mean // 2

    # 计算时间偏移量
    time_offset = peak_index_mean - (len(seq_1) - 1)

    return time_offset

def time_align(ref_signal_1, ref_signal_2):

    # # 时间偏移量估计
    # frame_bias = latency_estimate(ref_signal_1, ref_signal_2)  ### signal2 is real
    frame_bias = 0
    # 手动对齐
    plt.ioff()
    print('开始初始时间对齐')
    print('输入[+/-n]手动调整延迟量, 让sensor与mocap对齐,输入ok完成对齐')
    while True:
        print(f"当前传感器信号延迟量:{frame_bias}")
        if frame_bias < 0:
            plt.plot(ref_signal_1[:], label='sensor')
            plt.plot(ref_signal_2[-frame_bias:], label='mocap')
            plt.ylim(0, 50)
        else:
            plt.plot(ref_signal_1[frame_bias:], label='sensor')
            plt.plot(ref_signal_2[:], label='mocap')
            plt.ylim(0, 50)
        plt.legend()
        plt.show()
        c = input().lower()
        if c.find('+') > -1:
            frame_bias += int(c[1:])
        elif c.find('-') > -1:
            frame_bias -= int(c[1:])
        elif c.find('ok') > -1:
            # if frame_bias < 0:
            #     seq_3 = seq_3[-frame_bias:]
            #     seq_4 = seq_4[-frame_bias:]
            #     ref_signal_2 = ref_signal_2[-frame_bias:]
            # else:
            #     seq_1 = seq_1[frame_bias:]
            #     seq_2 = seq_2[frame_bias:]
            #     ref_signal_1 = ref_signal_1[frame_bias:]
            break

    return frame_bias

def read_imu_(imu_path):
    data = torch.load(imu_path)
    print(f'imu data loaded from {imu_path}, data keys: {data.keys()}')
    aM, RMB, pressure = data['acc'], data['ori'], data['pressure']
    return aM, RMB, pressure

def load_data(body_model, smpl_pose, fps=60):
    shape = torch.zeros(size=[10])
    tran = torch.zeros(size=[len(smpl_pose), 3])
    _, joint = body_model.forward_kinematics(smpl_pose, shape, tran, calc_mesh=False)
    left_hand_pos, right_hand_pos = joint[:, [20]], joint[:, [21]]

    left_hand_syn_acc = syn_acc(v=left_hand_pos, fps=fps)

    # 右手加速度作为校准参考信号
    left_hand_syn_acc_scale = torch.norm(left_hand_syn_acc, p=2, dim=-1)

    return left_hand_syn_acc_scale

def read_sensor_acc(data_path):
    data = torch.load(data_path)
    print(f'sensor data loaded from {data_path}, data keys: {data.keys()}')
    aM = data['acc'].cpu()
    return aM

def save_data(save_path, smpl_pose, smpl_tran, sensor_data_path, save_name='test.pt', frame_bias=0):
    print('saving data...')
    print(f'smpl_pose shape: {smpl_pose.shape}, smpl_tran shape: {smpl_tran.shape}')
    
    num_frames = smpl_pose.shape[0]
    
    keys_to_align = ['acc', 'raw_acc', 'ori', 'gyro', 'mag', 'pressure', 'ppg']
    data = torch.load(sensor_data_path)
    
    aligned_data = {}
    for key in data.keys():
        if key in keys_to_align:
            # align
            aligned_data[key] = data[key][frame_bias:]
            # crop
            aligned_data[key] = aligned_data[key][:num_frames]
        else:
            aligned_data[key] = data[key]
    
    # 新增pose_gt和tran_gt
    pose_gt = smpl_pose.clone()
    tran_gt = smpl_tran.clone()
    
    aligned_data['pose_gt'] = pose_gt
    aligned_data['tran_gt'] = tran_gt
    
    # print shape
    print(f'aligned_data keys: {aligned_data.keys()}')
    for key in aligned_data.keys():
        print(f'{key} shape: {aligned_data[key].shape}')
    
    torch.save(aligned_data, os.path.join(save_path, save_name))

if __name__ == "__main__":
    data_dir = './data/raw'
    output_dir = "./data/processed"
    
    smpl_dir = os.path.join(data_dir, 'smpl')
    sensor_dir = os.path.join(data_dir, 'sensor')
    
    subject = 'hyq_0320'
    sub_dir_output = os.path.join(output_dir, subject)
    os.makedirs(sub_dir_output, exist_ok=True)
    
    sub_dir = os.path.join(smpl_dir, subject)
    seq_names = os.listdir(sub_dir)
    seq_num = len(seq_names)

    print('len:', len(seq_names))
    body_model = art.ParametricModel(paths.smpl_file)
    
    for i in range(0, 1):
        print(f'Processing sequence {i+1}/{seq_num}...')
        
        # load sensor data
        data_path = os.path.join(sensor_dir, subject, str(i+1)+'.pt')
        acc = read_sensor_acc(data_path)
        
        # load easymocap smpl data
        seq_name = seq_names[i]
        seq_dir = os.path.join(sub_dir, seq_name)
        smpl_dir = os.path.join(seq_dir, 'smpl')
        seq_idx = seq_name.split('_')[-1]
        pose_name = 'smpl_pose_' + seq_idx + '.pt'
        tran_name = 'smpl_tran_' + seq_idx + '.pt'
        
        pose_path = os.path.join(smpl_dir, pose_name)
        tran_path = os.path.join(smpl_dir, tran_name)
        
        smpl_pose = torch.load(pose_path)
        smpl_tran = torch.load(tran_path)
        
        # region: time alignment
        left_hand_acc_smpl = load_data(body_model, smpl_pose, fps=30)
        left_hand_acc_sensor = torch.norm(acc[:, 0], p=2, dim=-1)
        
        frame_bias = time_align(ref_signal_1=left_hand_acc_sensor, ref_signal_2=left_hand_acc_smpl)
        # endregion
        
        save_data(sub_dir_output, smpl_pose, smpl_tran, data_path, save_name=str(i+1)+'.pt', frame_bias=frame_bias)
