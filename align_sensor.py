import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from config import sensor

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

if __name__ == "__main__":
    # data_path
    sensor_dir = "data/raw/sensor"
    subject = "hyq_0327"
    sub_dir = os.path.join(sensor_dir, subject)
    
    seq_nums = len(os.listdir(sub_dir))
    
    devices = ["Watch_left", "Watch_right", "Phone_left", "Phone_right", "Headset", "STag_left", "STag_right"]
    # devices = ["Watch_left", "Phone_left", "Headset"]
    
    for i in range(seq_nums):
        seq_name = f"seq_{i+1:02d}"
        print(f"Processing {seq_name}...")
        seq_dir = os.path.join(sub_dir, seq_name)
        
        acc_df_list = []
        start_index_list = []
        
        for device in devices:
            device_dir = os.path.join(seq_dir, device)
            
            # 依次可视化每个设备的加速度模长
            acc_file = find_csv_by_keywords(device_dir, ["acc"])
            print(f"  Device: {device}, Acc file: {acc_file}")
            df = pd.read_csv(acc_file)
            
            # plot_acc_magnitude(df, seq=seq_name, device=device)
            
            # 获取前多少帧的加速度峰值最大值
            max_acc_index = get_max_acc_index(df, n=3000)
            start_index_list.append(max_acc_index)
            
            # 去掉前max_acc_index帧
            df_aligned = df.iloc[max_acc_index-400:].reset_index(drop=True)
            acc_df_list.append(df_aligned)
            
        # # # plot all the aligned acc magnitude
        # plot_all_device_acc_magnitudes(acc_df_list, seq_name, devices)