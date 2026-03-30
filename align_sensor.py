import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from config import sensor
import articulate as art

# 可视化加速度模长
def plot_acc_magnitude(df, seq="seq_01", device="device"):
    acc_cols = [col for col in df.columns if 'acc' in col.lower() and 'time' not in col.lower()]

    acc_values = df[acc_cols].apply(pd.to_numeric, errors='coerce').to_numpy()
    acc_norm = np.linalg.norm(acc_values, axis=1) - 9.8
    
    # show figure
    plt.figure(figsize=(10, 4))
    plt.plot(acc_norm, label="Acceleration Magnitude")
    plt.title("{} - {}".format(seq, device))
    plt.xlabel("Sample Index")
    plt.ylabel("Acceleration Magnitude")
    plt.legend()
    plt.grid()
    plt.show()

def plot_all_device_acc_magnitudes(df_list, seq_name, devices):
    """
    可视化所有设备的加速度模长在同一张图上
    :param df_list: 各个设备的对齐后的加速度数据
    :param seq_name: 序列名称，用于图例或标题
    :param devices: 设备名称列表，用于图例
    """
    plt.figure(figsize=(10, 6))

    # 遍历所有设备并绘制加速度模长
    for i, df in enumerate(df_list):
        acc_cols = [col for col in df.columns if 'acc' in col.lower() and 'time' not in col.lower()]
        acc_values = df[acc_cols].apply(pd.to_numeric, errors='coerce').to_numpy()
        
        acc_norm = np.linalg.norm(acc_values, axis=1) - 9.8 + 20 * i # 加上偏移量以区分不同设备的曲线

        plt.plot(acc_norm, label=f"Device: {devices[i]}")

    plt.title(f"Acceleration Magnitude for {seq_name}")
    plt.xlabel("Sample Index")
    plt.ylabel("Acceleration Magnitude (m/s²)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def get_max_acc_index(df, n=3000):
    acc_cols = [col for col in df.columns if 'acc' in col.lower() and 'time' not in col.lower()]

    acc_values = df[acc_cols].apply(pd.to_numeric, errors='coerce').to_numpy()
    acc_norm = np.linalg.norm(acc_values, axis=1) - 9.8
    
    acc_norm_subset = acc_norm[:n]
    max_acc_index = np.argmax(acc_norm_subset)
    return max_acc_index

def find_csv_by_keywords(folder, keywords):
    """
    在 folder 下寻找文件名同时包含所有 keywords 的 csv 文件
    """
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".csv"):
            continue
        lower_name = fname.lower()
        if all(k.lower() in lower_name for k in keywords):
            return os.path.join(folder, fname)
    return None

def downsample(tensor, original_fps=100, target_fps=30):
    """
    使用线性插值将数据从 original_fps 下采样到 target_fps。
    :param tensor: 输入张量，形状为 (帧数, N)
    :param original_fps: 原始帧率
    :param target_fps: 目标帧率
    :return: 下采样后的张量
    """
    # 计算目标帧数
    original_len = tensor.shape[0]
    target_len = int(original_len * target_fps / original_fps)

    # 计算目标序号
    target_indices = np.linspace(0, original_len - 1, target_len)
    
    # 直接选取最近邻的帧进行下采样
    downsampled_tensor = tensor[target_indices.astype(int)]

    return downsampled_tensor

if __name__ == "__main__":
    # data_path
    sensor_dir = "data/raw/sensor"
    subject = "hyq_0327"
    sub_dir = os.path.join(sensor_dir, subject)
    
    seq_nums = len(os.listdir(sub_dir))
    
    devices = ["Watch_left", "Watch_right", "Phone_left", "Phone_right", "Headset", "STag_left", "STag_right"]
    
    for i in range(0, seq_nums):
        seq_name = f"seq_{i+1:02d}"
        print(f"Processing {seq_name}...")
        seq_dir = os.path.join(sub_dir, seq_name)
        
        sensor_data = {}
        accs, gyros, mags, quats, linear_accs, ppgs = [], [], [], [], [], []
        
        # 加载calibration矩阵
        cali_file = [f for f in os.listdir(seq_dir) if f.startswith(f"{i+1}_") and f.endswith(".pt")][0]
        cali_path = os.path.join(seq_dir, cali_file)
        
        cali_data = torch.load(cali_path)
        RMI, RSB = cali_data['RMI'], cali_data['RSB']
        
        for i, device in enumerate(devices):
            device_dir = os.path.join(seq_dir, device)
            
            # 获取前多少帧的加速度峰值最大值
            acc_file = find_csv_by_keywords(device_dir, ["acc"])
            acc_df = pd.read_csv(acc_file)
            
            # 计算加速度模长并找到峰值最大的位置
            max_acc_index = get_max_acc_index(acc_df, n=3000)
            
            keys = sensor.keys_in_device[device.split("_")[0]] # 根据设备类型获取对应的keys
            for key in keys:
                csv_file = find_csv_by_keywords(device_dir, [key])
                data_df = pd.read_csv(csv_file)

                # TODO: 提取对齐后的数据并保存到对应的列表中
                # 1. align data_df based on max_acc_index
                data_df = data_df.iloc[max_acc_index+300:].reset_index(drop=True)
                
                # 2. save aligned data_df to corresponding list based on key
                if 'acc' in key.lower() and 'line_acc' not in key.lower() and 'linear_acc' not in key.lower():
                    # 取后三列作为通道保存，保存为numpy形式
                    accs.append(data_df.iloc[:, -3:].to_numpy())
                elif 'gyro' in key.lower():
                    gyros.append(data_df.iloc[:, -3:].to_numpy())
                elif 'mag' in key.lower() or 'magnetic' in key.lower():
                    mags.append(data_df.iloc[:, -3:].to_numpy())
                elif 'quaternion' in key.lower() or 'rotation' in key.lower():
                    quats.append(data_df.iloc[:, -4:].to_numpy())
                elif 'line_acc' in key.lower() or 'linear_acc' in key.lower():
                    linear_accs.append(data_df.iloc[:, -3:].to_numpy())
                elif 'ppg' in key.lower():
                    ppgs.append(data_df.iloc[:, -11:].to_numpy())
            
        # 根据accs中序列的长度确定min_len
        min_len = min([acc.shape[0] for acc in accs])
        # 对齐所有数据到min_len
        accs = [acc[:min_len] for acc in accs]
        gyros = [gyro[:min_len] for gyro in gyros]
        mags = [mag[:min_len] for mag in mags]
        quats = [quat[:min_len] for quat in quats]
        linear_accs = [la[:min_len] for la in linear_accs]
        ppgs = [ppg[:min_len] for ppg in ppgs]
        # 将accs从[[N, 3], [N, 3], ...]转换为[N, device_num, 3]
        accs = torch.from_numpy(np.stack(accs, axis=1)).float()
        gyros = torch.from_numpy(np.stack(gyros, axis=1)).float()
        mags = torch.from_numpy(np.stack(mags, axis=1)).float()
        quats = torch.from_numpy(np.stack(quats, axis=1)).float()
        linear_accs = torch.from_numpy(np.stack(linear_accs, axis=1)).float()
        ppgs = torch.from_numpy(np.stack(ppgs, axis=1)).float()
        
        # apply calibration matrix
        aS = accs.clone()
        quat_IS = quats.clone()
        RIS = art.math.quaternion_to_rotation_matrix(quat_IS).view(-1, aS.shape[1], 3, 3)
        
        aI  = RIS.matmul(aS.unsqueeze(-1)).squeeze(-1) + torch.tensor([0, 0, - 9.8])
        aM  = RMI.matmul(aI.unsqueeze(-1)).squeeze(-1)
        RMB = RMI.matmul(RIS).matmul(RSB)
        print(f"aM shape: {aM.shape}, RMB shape: {RMB.shape}")
        
        aM = downsample(aM, original_fps=100, target_fps=30)
        RMB = downsample(RMB, original_fps=100, target_fps=30)
        accs = downsample(accs, original_fps=100, target_fps=30)
        gyros = downsample(gyros, original_fps=100, target_fps=30)
        mags = downsample(mags, original_fps=100, target_fps=30)
        quats = downsample(quats, original_fps=100, target_fps=30)
        linear_accs = downsample(linear_accs, original_fps=100, target_fps=30)
        ppgs = downsample(ppgs, original_fps=100, target_fps=30)
        
        # 确保每个模态的帧数一致
        assert aM.shape[0] == RMB.shape[0] == accs.shape[0] == gyros.shape[0] == mags.shape[0] == quats.shape[0] == linear_accs.shape[0] == ppgs.shape[0], "帧数不一致！"
        
        # 将对齐后的数据保存到sensor_data字典中
        sensor_data['acc'] = accs
        sensor_data['gyro'] = gyros
        sensor_data['mag'] = mags
        sensor_data['quaternion'] = quats
        sensor_data['linear_acc'] = linear_accs
        sensor_data['ppg'] = ppgs
        
        sensor_data['aM'] = aM
        sensor_data['RMB'] = RMB
        
        # save sensor_data as .pt file
        save_path = os.path.join(seq_dir, "sensor_data.pt")
        torch.save(sensor_data, save_path)