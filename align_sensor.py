import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
pd.options.mode.chained_assignment = None

def filter_dataframe(df, time_col, threshold):
    if time_col not in df.columns:
        print(f"警告：列 {time_col} 不存在，跳过筛选")
        return df
    return df[df[time_col] > threshold]

# 将毫秒时间戳转换为北京时间（UTC+8）
def timestamp_to_bj_time(timestamp_ms):
    # 转为 datetime 对象（从 1970-01-01 UTC 开始）
    dt_utc = datetime(1970, 1, 1) + timedelta(milliseconds=int(timestamp_ms))
    # 转换为北京时间（UTC+8）
    dt_bj = dt_utc + timedelta(hours=8)
    # 格式化为：2026/03/16/11:25:36.632
    return dt_bj.strftime("%Y/%m/%d/%H:%M:%S.%f")[:-3]

def unify_time_axis(
    variables_dict: dict,
    reference_device: str = "device_01",
    step_ms: int = 10,
):
    """
    统一所有设备的 AccTime 时间轴，使其从同一起点 m 开始，每行递增 step_ms。

    参数:
        variables_dict: batch_align_sensors 返回的字典
        reference_device: 参考设备名（默认 device1）
        time_col: 时间列名（默认 "AccTime"）
        step_ms: 时间步长（默认 10ms）
        verbose: 是否打印日志

    返回:
        unified_dict: 修改后的 variables_dict，所有设备 AccTime 已对齐
    """
    # 1. 检查参考设备是否存在
    if reference_device not in variables_dict:
        raise ValueError(f"参考设备 '{reference_device}' 不存在于 variables_dict 中！")

    ref_data = variables_dict[reference_device]
    if f"{reference_device}_ACC_df_resam" not in ref_data:
        raise ValueError(f"设备 {reference_device} 中未找到 'ACC_df_resam' 数据！")

    acc_df_resam = ref_data[f"{reference_device}_ACC_df_resam"]
    gyro_df_resam = ref_data[f"{reference_device}_GYRO_df_resam"]

    if "AccTime" not in acc_df_resam.columns:
        raise ValueError(f"时间列 AccTime 不存在于 {reference_device}_acc_df_resam 中！")

    # 2. 提取参考设备的 AccTime 列
    acc_time_series = acc_df_resam["AccTime"].values

    # 3. 找到第一行的值 t0
    t0 = int(acc_time_series[0])

    # 4. 找出所有与 t0 相同的行数 n
    mask = (acc_time_series == t0)
    matches = np.where(mask)[0] # 考虑浮点误差虑浮点误差
    n = len(matches)

    if n == 0:
        raise ValueError(f"未找到与第一行值 {t0} 匹配的行！")

    # 5. 计算新起始值 m
    m = t0 - step_ms * (n - 1)

    # 6. 为参考设备生成新的 AccTime 序列
    new_time_ref = np.arange(m, m + step_ms * len(acc_time_series), step_ms)
    time_strings = np.array([timestamp_to_bj_time(ts) for ts in new_time_ref])

    acc_df_resam.loc[:, "UTC"] = time_strings
    gyro_df_resam.loc[:, "UTC"] = time_strings

    # 7. 为其他设备生成相同的时间轴
    for device_name, device_data in variables_dict.items():
        if device_name == reference_device:
            continue  # 已处理

        if f"{device_name}_ACC_df_resam" not in device_data:
            continue

        df1 = device_data[f"{device_name}_ACC_df_resam"]
        df2 = device_data[f"{device_name}_GYRO_df_resam"]

        if "AccTime" not in df1.columns:
            continue

        n_rows = len(df1)
        new_time = np.arange(m, m + step_ms * n_rows, step_ms)
        time_strings = np.array([timestamp_to_bj_time(ts) for ts in new_time])

        df1.loc[:,"UTC"] = time_strings
        df2.loc[:,"UTC"] = time_strings

    return variables_dict

def data_timestamp_alignment(data):
    """
        对加速度数据按时间戳进行对齐采样。

        """
    time_col = [col for col in data.columns if 'Time' in col][0]
    timestamp_imu = np.array(pd.to_numeric(data[time_col], errors='coerce').values.tolist())

    # 转为 numpy 数组，确保操作高效
    start_time_data = timestamp_imu[0]
    end_time_data = timestamp_imu[-1]

    start_time = start_time_data
    end_time = end_time_data

    current_start = start_time
    resample_index = []

    # 逐个处理每个 100ms 的时间窗口
    while current_start <= end_time:
        current_end = current_start + 100  # 当前窗口结束时间

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

    return data.iloc[resample_index].reset_index(drop=True)

# ========================
# 配置参数
# ========================
# root_dir = r'D:\data\测试专用\260317'  # 根目录，根据实际情况修改
# output_root = r'D:\data\测试专用\aligned_output1'

root_dir = 'data_debug/debug/root'
output_root = 'data_debug/debug/output'
os.makedirs(output_root, exist_ok=True)

level_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

# 假设一级目录是 dir1, dir2, dir3
# 或者你可以手动指定：level_dirs = ['dir1', 'dir2', 'dir3']

if len(level_dirs) != 2:
    raise ValueError("根目录下必须恰好有2个一级目录")

# ========================
# 保存结果的变量
# ========================
acc_total1, acc_total2 = None, None
gyro_total1, gyro_total2 = None, None

# 用于存储每个txt的第一行数据（转为float）
txt_values = []

# ========================
# 处理每个一级目录
# ========================
for idx, dir_name in enumerate(level_dirs):
    dir_path = os.path.join(root_dir, dir_name)

    # 检查 ACC 和 GYRO 目录是否存在
    acc_dir = os.path.join(dir_path, 'ACC')
    gyro_dir = os.path.join(dir_path, 'GYRO')

    if not os.path.exists(acc_dir) or not os.path.exists(gyro_dir):
        raise FileNotFoundError(f"在 {dir_name} 中未找到 ACC 或 GYRO 目录")

    # 读取 ACC 目录下的所有 CSV
    acc_files = [f for f in os.listdir(acc_dir) if f.endswith('.csv')]
    acc_dfs = []
    for file in acc_files:
        file_path = os.path.join(acc_dir, file)
        df = pd.read_csv(file_path)
        # 提取文件名中的数字（split('_')[-2]）
        try:
            # sort_key = int(file.split('_')[-2].split('.')[0])
            sort_key = int(file.split('_')[0])
        except (ValueError, IndexError):
            sort_key = 0  # 若无法解析，则默认为0
        acc_dfs.append((sort_key, df))

    # 按 sort_key 升序排序
    acc_dfs.sort(key=lambda x: x[0])
    acc_df = pd.concat([df for _, df in acc_dfs], ignore_index=True)
    txt_values.append(acc_df.iloc[0]['AccTime'])

    # 读取 GYRO 目录下的所有 CSV
    gyro_files = [f for f in os.listdir(gyro_dir) if f.endswith('.csv')]
    gyro_dfs = []
    for file in gyro_files:
        file_path = os.path.join(gyro_dir, file)
        df = pd.read_csv(file_path)
        try:
            sort_key = int(file.split('_')[-2].split('.')[0])
        except (ValueError, IndexError):
            sort_key = 0
        gyro_dfs.append((sort_key, df))

    gyro_dfs.sort(key=lambda x: x[0])
    gyro_df = pd.concat([df for _, df in gyro_dfs], ignore_index=True)

    # 存入全局变量
    if idx == 0:
        acc_total1 = acc_df
        gyro_total1 = gyro_df
    elif idx == 1:
        acc_total2 = acc_df
        gyro_total2 = gyro_df

# ========================
# 找出三个 txt 第一行的最大值
# ========================
max_time = max(txt_values)

variables_dict = {}

device_01 = [acc_total1, gyro_total1]
device_02 = [acc_total2, gyro_total2]

for idx, device in enumerate([device_01, device_02]):
    device_dict = {}
    device_name = f'device_0{idx + 1}'
    sensor_type = ['ACC', 'GYRO']
    for sensor in sensor_type:
        if sensor == 'ACC':
            var_name = f"{device_name}_{sensor}_df"
            align_df = data_timestamp_alignment(device[0])
            filter_acc = filter_dataframe(align_df, time_col='AccTime', threshold=max_time)
            device_dict[var_name] = filter_acc
            last_col = filter_acc.columns[-1]
            subset = filter_acc.loc[:999, last_col]
            max_row_index = subset.idxmax()
            device_dict["ACC_MAX"] = max_row_index
            resample_set = filter_acc.loc[max_row_index:, :]
            resample_set.iloc[:, -3:] = resample_set.iloc[:, -3:] * 0.00024414
            # change q2: why multiply 0.00024414? 
            device_dict[f"{var_name}_resam"] = resample_set
        elif sensor == 'GYRO':
            var_name = f"{device_name}_{sensor}_df"
            align_df = data_timestamp_alignment(device[1])
            filter_gyro = filter_dataframe(align_df, time_col='GyroTime', threshold=max_time)
            device_dict[var_name] = filter_gyro
            resample_set = filter_gyro.loc[device_dict["ACC_MAX"]:,:]
            resample_set.iloc[:, -3:] = resample_set.iloc[:, -3:] * 0.00122173
            # change q3: why multiply 0.00122173?
            device_dict[f"{var_name}_resam"] = resample_set
    variables_dict[device_name] = device_dict

# ========================
unified_data = unify_time_axis(variables_dict= variables_dict, reference_device="device_01", step_ms=10,)
for device_name, data_dict in unified_data.items():
    # 获取对应设备的 acc_df_resam
    acc_df = data_dict.get(f"{device_name}_ACC_df_resam")
    gyro_df = data_dict.get(f"{device_name}_GYRO_df_resam")
    # 构造文件名
    filename = f"{device_name}_加速度数据.xlsx"
    filename1 = f'{device_name}_陀螺仪数据.xlsx'
    # 保存为 Excel
    acc_df.to_excel(os.path.join(output_root, filename), index=False)
    gyro_df.to_excel(os.path.join(output_root, filename1), index=False)