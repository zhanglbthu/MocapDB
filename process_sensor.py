# 这是一个用于统计传感器数据，对齐不同设备时间戳的脚本
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import sensor

def get_sorted_files(data_dir):
    # 获取所有以.pt结尾的文件
    pt_files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
    
    # 按照文件名前的数字部分排序
    pt_files.sort(key=lambda f: int(f.split('.')[0]))  # 通过文件名前的数字进行排序
    
    return pt_files

def print_duration(data_dir):
    
    total_frames = 0
    total_duration = 0.0  # in seconds

    # 获取data_dir下所有的.pt文件
    pt_files = get_sorted_files(data_dir)
    
    for pt_file in pt_files:
        file_path = os.path.join(data_dir, pt_file)
        
        # 加载.pt文件数据
        data = torch.load(file_path)
        
        # 假设每个.pt文件包含'acc'或类似的数据
        acc_data = data.get('acc', None)
        
        if acc_data is not None:
            # 获取帧数，'acc'数据是形状为[N, 3]的张量
            frames = acc_data.size(0)  # 取第一维，即帧数
            total_frames += frames
            
            # 计算时长，帧率为30fps
            duration = frames / 30.0  # 时长 = 帧数 / 帧率
            total_duration += duration
    
    print(f"Total frames: {total_frames}")
    print(f"Total duration: {total_duration} seconds")

def get_key_data(data_dir, key):
    key_data = []
    
    # 获取data_dir下所有的.pt文件
    pt_files = get_sorted_files(data_dir)
    
    for pt_file in pt_files:
        file_path = os.path.join(data_dir, pt_file)
        
        # 加载.pt文件数据
        data = torch.load(file_path)
        
        # 获取指定key的数据
        if key in data:
            key_data.append(data[key])
    
    return key_data

def plot_acc(data_dir, file_index=0, start_frame=0, end_frame=1000, devices_to_plot=None, time_offsets=None):
    # 获取加速度数据
    accs = get_key_data(data_dir, 'acc')
    
    # 提取指定文件的加速度数据
    acc = accs[file_index]  # acc: [num_frames, num_devices, 3]
    
    # 只取[start_frame, end_frame]区间的帧数据
    acc = acc[start_frame:end_frame]
    
    # 计算每帧的加速度模长
    acc_magnitude = torch.norm(acc, dim=2)  # 计算每帧每个设备的加速度模长 [num_frames, num_devices]
    
    # 如果用户没有指定设备列表，默认绘制所有设备
    if devices_to_plot is None:
        devices_to_plot = np.arange(acc.shape[1])  # 默认绘制所有设备
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 绘制每个设备的加速度模长曲线
    for device in devices_to_plot:
        if device < acc.shape[1]:  # 确保设备编号在有效范围内
            # 获取设备与设备0的时间偏差
            offset = time_offsets[device] if time_offsets is not None and device < len(time_offsets) else 0
            
            # 根据偏差调整数据（使用torch.roll平移数据）
            adjusted_data = torch.roll(acc_magnitude[:, device], shifts=offset, dims=0)
            
            # 绘制调整后的设备加速度模长曲线
            plt.plot(np.arange(start_frame, end_frame), adjusted_data.cpu().numpy(), label=f"Device {device + 1} (Offset {offset})")
        else:
            print(f"Warning: Device index {device} is out of range for time_offsets list.")
    
    # 设置标签和标题
    plt.xlabel('Frame Index')
    plt.ylabel('Acceleration Magnitude')
    plt.title(f'Acceleration Magnitude for Selected Devices (File {file_index + 1}, Frames {start_frame} to {end_frame})')
    plt.legend()
    plt.grid(True)
    plt.show()

def align(data_dir, time_offsets):
    # 获取文件夹下所有.pt文件
    pt_files = get_sorted_files(data_dir)

    subject = os.path.basename(data_dir)
    save_dir = os.path.join("data/raw/sensor", subject)
    
    # 定义需要对齐的键列表
    keys_to_align = ['acc', 'raw_acc', 'ori', 'gyro', 'mag', 'pressure', 'ppg']
    
    # 遍历所有文件
    for pt_file in pt_files:
        file_path = os.path.join(data_dir, pt_file)
        
        # 加载数据
        data = torch.load(file_path)
        
        aligned_data = {}
        
        # 获取RMI, RSB, acc_bias，直接保存，无需对齐
        aligned_data['RMI'] = data.get('RMI', None).clone()
        aligned_data['RSB'] = data.get('RSB', None).clone()
        aligned_data['acc_bias'] = data.get('acc_bias', None).clone()
        
        # 获取设备数量和帧数
        acc_data = data['acc']
        num_frames, num_devices, _ = acc_data.shape
        
        # 对每个键（数据类型），进行时间对齐
        for key in keys_to_align:
            if key in data:  # 确保数据中包含该键
                aligned_data[key] = data[key].clone()  # 直接使用key作为字典键，避免重复嵌套
                # 对每个设备，使用时间偏差进行滚动
                for device in range(1, num_devices):  # 跳过设备0
                    offset = time_offsets[device]
                    aligned_data[key][:, device, :] = torch.roll(
                        aligned_data[key][:, device, :], shifts=offset, dims=0
                    )
        
        # 计算正偏差和负偏差的最大值
        max_positive_offset = max([offset for offset in time_offsets if offset > 0], default=0)
        max_negative_offset = min([offset for offset in time_offsets if offset < 0], default=0)
        
        # 计算需要去除的帧数
        remove_start_frames = max_positive_offset
        remove_end_frames = abs(max_negative_offset)
        
        # 截取数据：去除不连贯的帧
        for key in keys_to_align:
            aligned_data[key] = aligned_data[key][remove_start_frames:num_frames-remove_end_frames]
        
        # 保存对齐后的数据，按原文件名保存
        aligned_data_path = os.path.join(save_dir, pt_file)
        torch.save(aligned_data, aligned_data_path)
        print(f"Aligned data saved to: {aligned_data_path}")

if __name__ == "__main__":
    # data_path
    sensor_dir = "data/raw/sensor_raw"
    subject = "hyq_0320"
    data_dir = os.path.join(sensor_dir, subject)
    
    '''
    data keys: raw_acc, acc, ori, gyro, mag, pressure, ppg, RMI, RSB, acc_bias, pose
    '''
    
    # align sensor data using sensor.time_offsets and save the results
    align(data_dir, sensor.time_offsets)
    