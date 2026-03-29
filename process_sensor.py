import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

def split_into_timestamp_packets(df, time_col):
    """
    按连续相同时间戳分包。
    返回一个列表，每个元素是:
    {
        "start": 起始行索引,
        "end": 结束行索引(开区间),
        "timestamp": 该包原始时间戳
    }
    """
    timestamps = pd.to_numeric(df[time_col], errors='coerce').to_numpy()
    n = len(timestamps)
    packets = []

    if n == 0:
        return packets

    start = 0
    for i in range(1, n):
        if timestamps[i] != timestamps[i - 1]:
            packets.append({
                "start": start,
                "end": i,
                "timestamp": timestamps[start]
            })
            start = i

    packets.append({
        "start": start,
        "end": n,
        "timestamp": timestamps[start]
    })

    return packets

def interpolate_packet_timestamps(df, time_col):
    """
    对共享包时间戳的数据，按照连续相同时间戳自动分包，
    再根据前后两包时间戳，为包内每条数据插值生成逐条时间戳。
    """
    df = df.copy()
    packets = split_into_timestamp_packets(df, time_col)
    n = len(df)

    if n == 0:
        return df

    new_timestamps = np.zeros(n, dtype=np.float64)

    if len(packets) == 1:
        # 只有一个包时，无法利用下一包时间戳，只能保持原值
        p = packets[0]
        new_timestamps[p["start"]:p["end"]] = p["timestamp"]
        df[time_col] = new_timestamps
        return df

    for i, packet in enumerate(packets):
        start = packet["start"]
        end = packet["end"]
        t0 = packet["timestamp"]
        packet_len = end - start

        if i < len(packets) - 1:
            t1 = packets[i + 1]["timestamp"]
            # 在 [t0, t1) 内均匀分布
            interp_ts = t0 + (np.arange(packet_len) / packet_len) * (t1 - t0)
        else:
            # 最后一包没有下一包，用前一包的间隔估计
            prev_t = packets[i - 1]["timestamp"]
            delta = t0 - prev_t
            interp_ts = t0 + (np.arange(packet_len) / packet_len) * delta

        new_timestamps[start:end] = interp_ts

    df[time_col] = np.round(new_timestamps).astype(np.int64)
    return df

def data_timestamp_alignment(data):
    """
    对加速度数据按时间戳进行对齐采样。

    """
    time_col = [col for col in data.columns if 'time' in col.lower()][0]
    timestamp_imu = np.array(pd.to_numeric(data[time_col], errors='coerce').values.tolist())

    # 转为 numpy 数组，确保操作高效
    start_time_data = timestamp_imu[0]
    end_time_data = timestamp_imu[-1]
    
    delta_time = end_time_data - start_time_data
    print(f"数据共持续 {delta_time} ms")

    start_time = start_time_data
    end_time = end_time_data

    current_start = start_time
    resample_index = []

    # 逐个处理每个 100ms 的时间窗口
    while current_start <= end_time:
        current_end = current_start + 100  # 
        
        # 找出在 [current_start, current_end] 区间内的所有索引
        # 注意：包含右边界（即 <= current_end）
        mask = (timestamp_imu >= current_start) & (timestamp_imu < current_end)
        valid_indices = np.where(mask)[0]

        if len(valid_indices) == 0:
            # 该窗口无数据，跳过
            sample_indices = np.linspace(start=stop_index, stop=stop_index, dtype=int, num=10, endpoint=False)
            # 保存这些采样索引
            resample_index.extend(sample_indices.tolist())
            # print(f"缺失{current_start}到{current_end}部分时间")
            current_start += 100
            continue

        # 取该窗口内的最大索引（即最后一个数据点）
        stop_index = valid_indices[-1]

        # 在当前窗口内，从 current_start 到 stop_index 之间进行采样
        # 使用 linspace 生成 10 个等间距的采样索引（整数）
        # 生成 100 个从 start_index 到 stop_index 的等间距索引（非包含终点）
        start_index = valid_indices[0]
        sample_indices = np.linspace(start=start_index, stop=stop_index+1, dtype=int, num=10, endpoint=False)

        # 保存这些采样索引
        resample_index.extend(sample_indices.tolist())

        # 下一个窗口
        current_start += 100

    # # 去除最后10条数据
    resample_index = resample_index[:-10]
    return data.iloc[resample_index].reset_index(drop=True)

if __name__ == "__main__":
    # data_path
    sensor_dir = "data/raw/sensor_raw"
    subject = "hyq_0327"
    data_dir = os.path.join(sensor_dir, subject)
    
    output_dir = "data/raw/sensor"
    save_dir = os.path.join(output_dir, subject)
    os.makedirs(save_dir, exist_ok=True)
    
    '''
    data_dir下有两个文件夹left和right: data_dir/left和data_dir/right, 分别存储了两侧的设备数据
    每个子文件夹下存储了每个设备的传感器数据，如./Headset, ./Phone, ...
    '''
    
    # 对于每个设备，拉齐不同模态的帧率和总帧数
    sides = ['left', 'right']
    
    devices_in_side = {'left': ['Headset', 'Phone', 'Watch', 'STag_C63', 'STag_D4D'],
                       'right': ['Phone', 'Watch']}
    
    # 每个设备需要保存的keys
    keys_in_device = {'Headset':  ['acc_headsetL', 'gyro_headsetL', 'quaternion_left'],
                      'Phone':    ['acc', 'gyro', 'linear_acc', 'magnetic', 'rotation'],
                      'Watch':    ['acc', 'gyro', 'line_acc', 'mag', 'ppg', 'quaternion'],
                      'STag_C63': ['acc', 'gyro', 'quaternion'],
                      'STag_D4D': ['acc', 'gyro', 'quaternion']}
    
    # 每个设备需要进行时间戳插值的keys
    keys_to_interpolate = {'Headset':  ['acc_headsetL', 'gyro_headsetL'],
                           'Phone':    [],
                           'Watch':    ['acc', 'gyro', 'line_acc', 'mag', 'ppg'],
                           'STag_C63': ['acc', 'gyro'],
                           'STag_D4D': ['acc', 'gyro']}
    
    for side in sides:
        side_dir = os.path.join(data_dir, side)
        seq_names = os.listdir(side_dir)
        for i, seq_name in enumerate(seq_names):
            seq_dir = os.path.join(side_dir, seq_name)
            device_names = devices_in_side[side]
            for device_name in device_names:
                print(f"Processing {device_name} in {seq_name} ({side})...")
                device_dir = os.path.join(seq_dir, device_name)
                if not os.path.exists(device_dir):
                    print(f"Warning: {device_dir} does not exist.")
                    continue
                
                keys_to_align = keys_in_device[device_name]
                df_list, time_col_list = [], []
                # TODO: TimeStamp Interpolation for keys in keys_to_interpolate[device_name]
                for key in keys_to_align:
                    csv_file = find_csv_by_keywords(device_dir, [key])
                    if csv_file is None:
                        print(f"Warning: No CSV file found for {device_name} with key '{key}' in {device_dir}.")
                        continue
                    
                    if 'rotation' in key.lower():
                        # 跳过错误表头，重新读
                        df = pd.read_csv(csv_file, header=None, skiprows=1)
                        df.columns = ['time', 'w', 'x', 'y', 'z']
                    else:
                        df = pd.read_csv(csv_file)
                        
                    time_col = [col for col in df.columns if 'time' in col.lower()][0]
                    
                    # 处理特殊情况，如果key是baro，则原地把时间戳从16位改成13位，抹去后面的三个0
                    if 'baro' in key.lower():
                        df[time_col] = (pd.to_numeric(df[time_col], errors='coerce') // 1000).astype(np.int64)
                    
                    # 对于需要插值的key，进行时间戳插值
                    if key in keys_to_interpolate[device_name]:
                        df_interp = interpolate_packet_timestamps(df, time_col)
                        df_list.append(df_interp)
                    else:
                        df_list.append(df)
                        
                    time_col_list.append(time_col)
                
                assert len(df_list) == len(keys_to_align), f"Expected {len(keys_to_align)} dataframes for {device_name}, but got {len(df_list)}."
                
                # 拉齐时间戳并采样至100fps
                # 获取最短的时间范围，裁剪数据
                min_time = max(df[time_col].min() for df, time_col in zip(df_list, time_col_list))
                max_time = min(df[time_col].max() for df, time_col in zip(df_list, time_col_list))
                
                df_list = [df[(df[time_col] >= min_time) & (df[time_col] <= max_time)].reset_index(drop=True) for df, time_col in zip(df_list, time_col_list)]
                
                if device_name != 'Headset':
                    df_list = [data_timestamp_alignment(df) for df in df_list]
        
                min_frames = min(len(df) for df in df_list)
                df_list = [df.iloc[:min_frames].reset_index(drop=True) for df in df_list]
                
                # save the aligned dataframes to new csv files
                # save_device_name = device_name + side
                save_seq_name = 'seq_' + str(i+1).zfill(2)
                if device_name == 'STag_C63':
                    save_device_name = 'STag_left'
                elif device_name == 'STag_D4D':
                    save_device_name = 'STag_right'
                else:
                    save_device_name = device_name + '_' + side if device_name != 'Headset' else device_name
                save_device_dir = os.path.join(save_dir, save_seq_name, save_device_name)
                os.makedirs(save_device_dir, exist_ok=True)
                for df, key in zip(df_list, keys_to_align):
                    original_csv = find_csv_by_keywords(device_dir, [key])
                    if original_csv is None:
                        continue
                    save_path = os.path.join(save_device_dir, os.path.basename(original_csv))
                    df.to_csv(save_path, index=False)