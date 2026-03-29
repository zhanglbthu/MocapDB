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

def syn_acc(v, smooth_n=2, fps=60):
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

def bvh_trans_downsampled(bvh_path, ly_path, xrs_path, fps_set=60):
    # bvh_smpl_map = [[0, 1, 2, 3, 4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    #                 [0, 3, 6, 9, 13, 16, 18, 20, 14, 17, 19, 21, 12, 15, 1,  4,  7,  10, 2,  5,  8,  11, 22, 23]]
    
    # upper body    
    bvh_smpl_map = [[0, 1, 2, 3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
                    [0, 6, 9, 12, 15, 13, 16, 18, 20, 14, 17, 19, 21, 1,  2,  3,  4,  5,  7,  8,  10, 11, 22, 23]]
    
    # 提取Frame#和Timestamp列
    frames = []
    timestamps = []
    hips = {'x': [], 'y': [], 'z': []}

    with open(ly_path, 'r') as file:
        lines = file.readlines()
    # 假设文件头占据前6行，调整此值以确保从正确位置开始读取数据
    data_start_index = 6
    for line in lines[data_start_index:]:
        parts = line.split()
        if len(parts) >= 3:  # 确保行包含足够的列
            frame = int(parts[0])
            timestamp = int(parts[2])
            frames.append(frame)
            timestamps.append(timestamp)

            parts_length = len(parts)  # 检查

    with open(xrs_path, 'r') as file:
        xrs_lines = file.readlines()

    # 文件头占据前32行，调整此值以确保从正确位置开始读取数据
    # data_start_index = 32 # whole body
    data_start_index = 23 # upper body

    # 用于记录第0帧的hips的x和z坐标值，初始化为None
    hips_0_x = None
    hips_0_z = None

    for index, xrs_line in enumerate(xrs_lines[data_start_index:], start=data_start_index):
        xrs_parts = xrs_line.split()
        if len(xrs_parts) >= 5:  # 确保行包含足够的列--hips中的toParent和toGlobal位置信息一致
            hips_x = float(xrs_parts[2])
            hips_y = float(xrs_parts[3])
            hips_z = float(xrs_parts[4])

            if index == data_start_index:
                # 如果是第0帧（首次进入循环），记录其x和z坐标值作为第0帧位置的对应坐标
                hips_0_x = hips_x
                hips_0_y = hips_y
                hips_0_z = hips_z
                # 将各个点的坐标数据添加到对应的字典列表中（第0帧正常添加，后续帧会基于此做归零处理）
                hips['x'].append(0)
                hips['y'].append(0)
                hips['z'].append(0)
            else:
                # 对于非第0帧的数据，将当前帧的x坐标减去第0帧的xyz坐标进行归零处理
                adjusted_hips_x = hips_x - hips_0_x
                adjusted_hips_y = hips_y - hips_0_y
                adjusted_hips_z = hips_z - hips_0_z
                # 将归零后的坐标数据添加到对应的字典列表中
                hips['x'].append(adjusted_hips_x)
                hips['y'].append(adjusted_hips_y)
                hips['z'].append(adjusted_hips_z)
    
    with open(bvh_path, 'r') as file:
        bvh_data = file.readlines()

    # 解析HIERARCHY部分和MOTION部分
    hierarchy_data = []
    motion_data = []
    is_motion_section = False

    for line in bvh_data:
        if line.strip() == "MOTION":
            is_motion_section = True
        if is_motion_section:
            motion_data.append(line.strip())
        else:
            hierarchy_data.append(line.strip())

    # 提取帧时间
    frame_time_line = motion_data[2]
    frame_time = float(frame_time_line.split(":")[1].strip())

    # 提取角度数据
    motion_lines = motion_data[3:]  # 跳过"Frames:", "Frame Time:"行和空行
    channels_data = []
    
    for line in motion_lines:
        channels = list(map(float, line.split()))
        # * upper body: len(channels) == 78; whole body: len(channels) == 132
        channels_data.append(channels)
    
    # # whole body
    # angle_data = [[sublist[i:i + 3] for i in range(3, 132, 6)] for sublist in channels_data]
    # pose_bvh = torch.tensor(np.array(angle_data, dtype='float')).reshape(-1, 22, 3)
    
    # upper body
    # init pose_bvh with zeros, and then fill the upper body joints with the corresponding angle data
    pose_bvh = torch.zeros(size=[len(channels_data), 22, 3])
    angle_data = [[sublist[i:i + 3] for i in range(3, 78, 6)] for sublist in channels_data]
    pose_bvh[:, :13, :] = torch.tensor(np.array(angle_data, dtype='float')).reshape(-1, 13, 3)  
    
    zeros = torch.zeros(pose_bvh.size(0), 2, 3)
    pose_bvh = torch.cat((pose_bvh, zeros), dim=1).reshape(-1, 24, 3) * np.pi / 180

    pose_smpl = torch.zeros(size=[pose_bvh.shape[0], 24, 3])
    pose_smpl[:, bvh_smpl_map[1]] += pose_bvh[:, bvh_smpl_map[0]]

    pose_bvh_matrix = art.math.euler_angle_to_rotation_matrix(pose_smpl.view(-1, 3), seq='ZXY').view(-1, 24, 3,
                                                                                                     3)  ###(6w,3,3)
    timestamps_bvh = [timestamps[0] + i * frame_time for i in range(frames[-1])]
    original_time_stamps = timestamps_bvh
    original_frame_numbers = list(range(len(original_time_stamps)))
    
    original_angles = rotation_matrix_to_r6d(pose_bvh_matrix).reshape(len(angle_data), -1)
    original_angles = original_angles.reshape(original_angles.shape[0], -1)
    # 计算原始帧率为 90fps 时，每个数据点的时间间隔
    original_frame_interval = 1 / 90
    # 目标帧率为 60fps，计算目标时间间隔
    target_frame_interval = 1 / fps_set
    # 确定新的时间戳数量
    num_original_frames = len(original_time_stamps)
    num_target_frames = int(num_original_frames * fps_set / 90)
    # 创建新的时间戳序列
    new_time_stamps = [original_time_stamps[0] + i * target_frame_interval for i in range(num_target_frames)]
    # 使用线性插值分别对帧序号和角度数据进行插值
    f_frame_number = interp1d(original_time_stamps, original_frame_numbers, kind='linear')
    interpolated_frame_numbers = f_frame_number(new_time_stamps)
    interpolated_angles = []
    for i in range(original_angles.shape[1]):
        f_angle_i = interp1d(original_time_stamps, original_angles[:, i], kind='linear')
        interpolated_angle_i = f_angle_i(new_time_stamps)
        interpolated_angles.append(interpolated_angle_i)
    interpolated_angles = np.array(interpolated_angles).T.tolist()

    # 对hips数据进行降采样（线性插值）
    interpolated_hips_x = []
    interpolated_hips_y = []
    interpolated_hips_z = []
    for coord in [hips['x'], hips['y'], hips['z']]:
        f_coord = interp1d(original_time_stamps, coord, kind='linear')
        interpolated_value = f_coord(new_time_stamps)
        if coord is hips['x']:
            interpolated_hips_x = interpolated_value
        elif coord is hips['y']:
            interpolated_hips_y = interpolated_value
        else:
            interpolated_hips_z = interpolated_value

    interpolated_hips_positions = list(zip(interpolated_hips_x, interpolated_hips_y, interpolated_hips_z))

    # 构建下采样后的新数据列表
    new_combined_data = []
    for ts, frame_number, angle in zip(new_time_stamps, interpolated_frame_numbers, interpolated_angles):
        new_combined_data.append((ts, angle))

    down_bvh = torch.tensor([t[1] for t in new_combined_data]).reshape(-1, 6)
    down_bvh = r6d_to_rotation_matrix(down_bvh).reshape(-1, 24, 3, 3)

    return down_bvh, interpolated_hips_positions  # interpolated_avg_waist_positions

def bvh_2_smpl_pose_down_sampled(bvh_path, ly_path, xrs_path, fps_set=30, tposeframe=1000):
    body_model_light = SMPLight()
    joint_num = 0
    joint_num = 22
    print(f'joint num: {joint_num}')
    # pose_bvh_matrix = bvh_downsampled(bvh_path=bvh_path, ly_path=ly_path,fps_set=fps_set)
    pose_bvh_matrix, tran_smpl = bvh_trans_downsampled(bvh_path=bvh_path, ly_path=ly_path, xrs_path=xrs_path,
                                                       fps_set=fps_set)
    # 第一帧是T-Pose 跳过

    # pose_bvh_matrix: [N, 24, 3, 3]
    # pose_bvh_matrix_g: [N, 24, 3, 3]
    pose_bvh_matrix_g = body_model_light.forward_kinematics(pose_bvh_matrix)

    # 设置t-pose帧位置
    t_pose_begin = tposeframe
    bvh_tpose = pose_bvh_matrix_g[t_pose_begin]
    
    # change1: not using bvh2smpl: niuqu
    # change2: not using transpose
    bvh2smpl = bvh_tpose.transpose(-2, -1)
    pose_smpl_g = pose_bvh_matrix_g.matmul(bvh2smpl)
    pose_smpl_l = body_model_light.inverse_kinematics(pose_smpl_g)
    pose_smpl = rotation_matrix_to_axis_angle(pose_smpl_l).reshape(-1, 24, 3)
    # print(pose_smpl)

    # 先将包含元组的列表转换为二维列表，再转换为张量
    tran_smpl_list = [list(item) for item in tran_smpl]
    tran_smpl_tensor = torch.tensor(tran_smpl_list)

    return pose_smpl, tran_smpl_tensor

def time_align_dynamic(ref_signal_1, ref_signal_2, seq_1, seq_2,judge_if_del_rate=1.01, cal_cc_window_len=120, plot_result=True,judge_del_per_frame=10):
    print('before delete, seq_1:', len(seq_1), 'mocap:',len(seq_2))
    index = 0
    # while False:
    while index+cal_cc_window_len < len(ref_signal_2) and index+cal_cc_window_len < len(ref_signal_1):
        # del_syn = True
        # del_imu = True

        cc = torch.corrcoef(
            torch.cat([ref_signal_2[index:index + cal_cc_window_len], ref_signal_1[index:index + cal_cc_window_len]],
                      dim=0).reshape(2, -1))[0, 1]

        del_seq_1 = True
        del_seq_2 = True

        while del_seq_2:
            del_seq_2_cc = torch.corrcoef(torch.cat([ref_signal_2[index + 1:index + 1 + cal_cc_window_len],
                                                   ref_signal_1[index:index + cal_cc_window_len]], dim=0).reshape(2,-1))[0, 1]
            if (cc > 0 and del_seq_2_cc > cc * judge_if_del_rate) or (cc < 0 and del_seq_2_cc > cc / judge_if_del_rate):
                cc = del_seq_2_cc
                del_seq_1 = False
                ref_signal_2=ref_signal_2[torch.arange(ref_signal_2.size(0)) != index]
                seq_2=seq_2[torch.arange(seq_2.size(0)) != index]
                # print('del imu at', index)
            else:
                del_seq_2 = False
        while del_seq_1:
            del_seq_1_cc = torch.corrcoef(
            torch.cat([ref_signal_2[index:index + cal_cc_window_len], ref_signal_1[index+1:index+1 + cal_cc_window_len]],
                      dim=0).reshape(2, -1))[0, 1]
            if (cc > 0 and del_seq_1_cc > cc * judge_if_del_rate) or (cc < 0 and del_seq_1_cc > cc / judge_if_del_rate):
                cc = del_seq_1_cc
                ref_signal_1=ref_signal_1[torch.arange(ref_signal_1.size(0)) != index]
                seq_1=seq_1[torch.arange(seq_1.size(0)) != index]
                del_seq_2 = False
                # print('del pose at', index)
            else:
                del_seq_1 = False
        index += judge_del_per_frame
        if cc < 0:
            print('woc')
    print('after delete, seq_1:', len(seq_1), ',mocap:',len(seq_2))
    if plot_result:
        plt.ioff()
        plt.plot(ref_signal_1, label='seq_1')
        plt.plot(ref_signal_2, label='seq_2')
        plt.legend()
        plt.show()


    data_len = min(len(seq_1), len(seq_2))
    seq_1 = seq_1[:data_len]
    seq_2 = seq_2[:data_len]

    print('\n时间对齐完成!')

    return seq_1, seq_2

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

    frame_bias = 0
    # 手动对齐
    plt.ioff()
    print('开始初始时间对齐')
    print('输入[+/-n]手动调整延迟量, 让sensor与mocap对齐,输入ok完成对齐')
    while True:
        print(f"当前传感器信号延迟量:{frame_bias}")
        if frame_bias < 0:
            plt.plot(ref_signal_1, label='xingying')
            plt.plot(ref_signal_2[-frame_bias:], label='easymocap')
            plt.ylim(0, 50)
        else:
            plt.plot(ref_signal_1[frame_bias:], label='xingying')
            plt.plot(ref_signal_2, label='easymocap')
            plt.ylim(0, 50)
        plt.legend()
        plt.show()
        c = input().lower()
        if c.find('+') > -1:
            frame_bias += int(c[1:])
        elif c.find('-') > -1:
            frame_bias -= int(c[1:])
        elif c.find('ok') > -1:
            break
    return frame_bias

def read_imu_(imu_path):
    data = torch.load(imu_path)
    print(f'imu data loaded from {imu_path}, data keys: {data.keys()}')
    aM, RMB, pressure = data['acc'], data['ori'], data['pressure']
    return aM, RMB, pressure

def read_gt(save_path, manual_gt_tpose_frame, fps_set):
    # 找到以.bvh和.xrs为后缀的唯一文件
    bvh_files = [f for f in os.listdir(save_path) if f.endswith('.bvh')]
    xrs_files = [f for f in os.listdir(save_path) if f.endswith('.xrs')]
    ly_files = [f for f in os.listdir(save_path) if f.endswith('.ly')]
    
    bvh_path = os.path.join(save_path, bvh_files[0]) 
    ly_path = os.path.join(save_path, ly_files[0])
    xrs_path = os.path.join(save_path, xrs_files[0])

    smpl_pose, smpl_tran = bvh_2_smpl_pose_down_sampled(bvh_path=bvh_path, ly_path=ly_path, xrs_path=xrs_path, fps_set=fps_set, tposeframe=manual_gt_tpose_frame)  ###(200w,60)
    smpl_pose = axis_angle_to_rotation_matrix(smpl_pose).reshape(-1, 24, 3, 3)
    print(smpl_pose.shape, smpl_tran.shape) ### 帧率为60
    return smpl_pose, smpl_tran

def load_data(body_model, smpl_pose, fps=60):
    shape = torch.zeros(size=[10])
    tran = torch.zeros(size=[len(smpl_pose), 3])
    _, joint = body_model.forward_kinematics(smpl_pose, shape, tran, calc_mesh=False)
    left_hand_pos, right_hand_pos = joint[:, [20]], joint[:, [21]]

    left_hand_syn_acc = syn_acc(v=left_hand_pos)

    # 右手加速度作为校准参考信号
    left_hand_syn_acc_scale = torch.norm(left_hand_syn_acc, p=2, dim=-1)

    return left_hand_syn_acc_scale

def save_data(save_path, smpl_pose, smpl_tran, smpl_pose_em, smpl_tran_em, save_name='test.pt'):
    print('saving data...')
    print(f'smpl_pose shape: {smpl_pose.shape}, smpl_tran shape: {smpl_tran.shape}, smpl_pose_em shape: {smpl_pose_em.shape}, smpl_tran_em shape: {smpl_tran_em.shape}')
    torch.save(
        {'pose': [smpl_pose.cpu()], 'tran': [smpl_tran / 1000], 'pose_em': [smpl_pose_em.cpu()], 'tran_em': [smpl_tran_em / 1000]},
        os.path.join(save_path, save_name))

if __name__ == "__main__":
    '''
    将xingying导出的数据和处理好的其他系统数据放在用同一个路径(imu_path)，
    对齐后得到的数据会保存在相同的路径
    '''
    data_dir = './data'
    em_dir = os.path.join(data_dir, 'processed')
    xy_dir = os.path.join(data_dir, 'raw', 'xingying')
    output_dir = os.path.join(data_dir, 'processed')
    sub_name = 'hyq_0320'
    sub_name_xy = 'hyq0320'
    
    sub_dir_output = os.path.join(output_dir, sub_name)
    os.makedirs(sub_dir_output, exist_ok=True)
    
    sub_dir_em = os.path.join(em_dir, sub_name)
    sub_dir_xy = os.path.join(xy_dir, sub_name)
    seq_names_em = os.listdir(sub_dir_em)
    seq_num = len(seq_names_em)
    
    print('len:', len(seq_names_em))
    body_model = art.ParametricModel(paths.smpl_file)
    
    keys_to_align = ['acc', 'raw_acc', 'ori', 'gyro', 'mag', 'pressure', 'ppg', 'pose', 'pose_gt']
    
    for i in range(10, len(seq_names_em)):
        print(f'Processing sequence {i+1}/{seq_num}...')
        
        # load xingying smpl data
        seq_name_xy = sub_name_xy + str(i+1)
        seq_dir_xy = os.path.join(sub_dir_xy, seq_name_xy)
        save_path = seq_dir_xy
        tpose_frame = 1
        fps_set = 30
        manual_gt_tpose_frame = int(tpose_frame * (3 / 9))
        smpl_pose, smpl_tran = read_gt(save_path, manual_gt_tpose_frame, fps_set)
        
        seq_name_em = str(i+1) + ".pt"
        data = torch.load(os.path.join(sub_dir_em, seq_name_em))
        
        smpl_pose_em = data['pose_gt']
        smpl_tran_em = data['tran_gt']
        
        # region: time alignment
        left_hand_syn_acc_scale_xy = load_data(body_model, smpl_pose)
        left_hand_syn_acc_scale_em = load_data(body_model, smpl_pose_em)
        frame_bias = time_align(ref_signal_1=left_hand_syn_acc_scale_xy, ref_signal_2=left_hand_syn_acc_scale_em)
        # endregion
        
        smpl_pose = smpl_pose[frame_bias:]
        
        if len(smpl_pose_em) < len(smpl_pose):
            smpl_pose = smpl_pose[:len(smpl_pose_em)]
        else:
            smpl_pose_em = smpl_pose_em[:len(smpl_pose)]
            print('warning: smpl_pose_em is shorter than smpl_pose, check the alignment result!')
            for key in data.keys():
                if key in keys_to_align:
                    data[key] = data[key][:len(smpl_pose)]
        
        # replace arms with smpl pose data
        pose_em = body_model.forward_kinematics(smpl_pose_em)[0]
        pose_xy = body_model.forward_kinematics(smpl_pose)[0]
        
        replace_joints = [13, 14, 16, 17, 18, 19, 20, 21]
        pose_em[:, replace_joints] = pose_xy[:, replace_joints]
        pose_local = body_model.inverse_kinematics_R(pose_em)
        
        data['pose_gt_new'] = pose_local
        
        torch.save(data, os.path.join(sub_dir_em, seq_name_em))
        
        print(f'data keys after refinement: {data.keys()}')
