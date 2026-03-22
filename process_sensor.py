# 这是一个用于统计传感器数据，对齐不同设备时间戳的脚本
import os
import torch

def print_duration(data_dir):
    
    total_frames = 0
    total_duration = 0.0  # in seconds

    # 获取data_dir下所有的.pt文件
    pt_files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
    
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

def rename_data_files(data_dir):
    # 获取所有以.pt结尾的文件
    pt_files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
    
    # 按照原文件名排序
    pt_files.sort()
    
    # 遍历文件并重命名
    for pt_file in pt_files:
        # 获取文件的完整路径
        old_path = os.path.join(data_dir, pt_file)
        
        idx = pt_file.split('_')[0]  # 获取文件名中的数字部分
        # 构造新的文件名，使用数字来命名
        new_filename = f"{idx}.pt"
        
        if pt_file == new_filename:
            print(f"Skipping: {old_path} (already named correctly)")
            continue
        
        new_path = os.path.join(data_dir, new_filename)
        
        # 重命名文件
        os.rename(old_path, new_path)
        print(f"Renamed: {old_path} -> {new_path}")

def get_key_data(data_dir, key):
    key_data = []
    
    # 获取data_dir下所有的.pt文件
    pt_files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
    
    for pt_file in pt_files:
        file_path = os.path.join(data_dir, pt_file)
        
        # 加载.pt文件数据
        data = torch.load(file_path)
        
        # 获取指定key的数据
        if key in data:
            key_data.append(data[key])
    
    return key_data

if __name__ == "__main__":
    # data_path
    sensor_dir = "data/raw/sensor"
    subject = "hyq_0320"
    data_dir = os.path.join(sensor_dir, subject)
    
    rename_data_files(data_dir)
    '''
    data keys: raw_acc, acc, ori, gyro, mag, pressure, ppg, RMI, RSB, acc_bias, pose
    '''
    
    acc = get_key_data(data_dir, 'acc')
    print(f"acc data count: {len(acc)}")
    
    