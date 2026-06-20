# MocapDB 数据处理 Pipeline

本文档说明当前 MocapDB 项目中从原始 calibration / sensor 数据，到多设备同步、sensor-SMPL 对齐、pair 数据生成和可视化质检视频的完整自动化流程。

## 一键运行

推荐在 Windows PowerShell 中运行：

```powershell
conda run -n baroposer python run_subject_pipeline.py --subject <subject>
```

例如：

```powershell
conda run -n baroposer python run_subject_pipeline.py --subject hyq_0402
conda run -n baroposer python run_subject_pipeline.py --subject yl_0403
conda run -n baroposer python run_subject_pipeline.py --subject lisha_0407
conda run -n baroposer python run_subject_pipeline.py --subject huohuo_0407
conda run -n baroposer python run_subject_pipeline.py --subject chaoran_0529
conda run -n baroposer python run_subject_pipeline.py --subject zhenhong_0529
```

也提供 bash 薄封装：

```bash
bash run_subject_pipeline.sh <subject>
```

Windows 下如果没有完整 WSL / bash / conda 环境，优先使用 PowerShell 版本的一键命令。

## 总控脚本行为

入口脚本：

```text
run_subject_pipeline.py
```

默认执行以下步骤：

1. 检查或提取 raw 数据。
2. 处理原始 sensor CSV，拉齐同设备内部时间戳和帧率。
3. 根据开头跳跃动作的加速度峰值进行多设备同步。
4. 自动对齐 sensor 与 SMPL。
5. 用模型推理 `pose_pred`，并生成 `pose_pred` / `pose_gt` side-by-side 质检视频。
6. 写出 pipeline summary。

如果 `data/raw` 中已经存在该 subject 的 raw 数据，脚本会直接复用，不重新提取：

```text
data/raw/sensor_raw/<subject>/left
data/raw/sensor_raw/<subject>/right
data/raw/calibration/<subject>
```

如果 raw 数据不完整，则默认从以下位置提取：

```text
E:\Research\project\daily_mocap\dataset\data\sensor\<date>\left
E:\Research\project\daily_mocap\dataset\data\sensor\<date>\right
E:\Research\project\daily_mocap\dataset\code\SmartWear\data\recording\<subject>
```

`date` 会优先从 calibration 文件名中自动推断；无法推断时可手动指定：

```powershell
conda run -n baroposer python run_subject_pipeline.py --subject <subject> --date 20260529
```

## 常用参数

跳过可视化视频：

```powershell
conda run -n baroposer python run_subject_pipeline.py --subject <subject> --skip-visualize
```

只重新生成视频，跳过前面步骤：

```powershell
conda run -n baroposer python run_subject_pipeline.py --subject <subject> --skip-process-sensor --skip-align-sensor --skip-align-smpl
```

只重新跑 sensor-SMPL 对齐和视频：

```powershell
conda run -n baroposer python run_subject_pipeline.py --subject <subject> --skip-process-sensor --skip-align-sensor
```

指定本地 sensor 源路径：

```powershell
conda run -n baroposer python run_subject_pipeline.py --subject <subject> --date 20260529 --left-sensor-root E:\path\left --right-sensor-root E:\path\right
```

指定设备顺序：

```powershell
conda run -n baroposer python run_subject_pipeline.py --subject <subject> --device-order Watch_left Watch_right Phone_left Phone_right Headset STag_left STag_right
```

## 目录结构

自动化流程主要读写以下目录：

```text
data/
  raw/
    calibration/<subject>/          # calibration .pt
    sensor_raw/<subject>/left/      # 左手机导出的原始 sensor 文件夹
    sensor_raw/<subject>/right/     # 右手机导出的原始 sensor 文件夹
    sensor/<subject>/seq_xx/        # process + sensor align 后的序列目录
    smpl/<subject>/smpl_*/          # EasyMocap / SMPL 结果
    manifests/<subject>/            # 每一步 manifest / summary
  auto_processed/<subject>/         # 最终 pair .pt
  auto_processed/<subject>_videos/  # pred-vs-gt mp4 质检视频
  alignment_reports/<subject>/      # sensor-SMPL 对齐报告图
```

## Step 1: raw 数据提取

脚本：

```text
tools/extract_raw_records.ps1
```

作用：

- 从 SmartWear recording 中读取 calibration。
- 从本地 sensor 源目录或手机 MTP 路径中筛选对应 sensor 记录。
- 根据 calibration 文件名中的时间与 sensor 文件夹时间进行匹配。
- 使用文件夹大小和 CSV 数量过滤明显废弃记录。
- 分 left / right 两侧复制到 `data/raw/sensor_raw/<subject>/`。
- 写出 `data/raw/manifests/<subject>/extract_left_<date>.json` 和 `extract_right_<date>.json`。

一键总控中会自动调用该脚本，但只有当 `data/raw` 中缺少 raw 数据时才会提取。

## Step 2: sensor CSV 预处理

脚本：

```text
process_sensor_auto.py
```

输入：

```text
data/raw/sensor_raw/<subject>/left/<timestamp>/
data/raw/sensor_raw/<subject>/right/<timestamp>/
data/raw/calibration/<subject>/*.pt
```

输出：

```text
data/raw/sensor/<subject>/seq_01/
data/raw/sensor/<subject>/seq_02/
...
```

功能：

- 将左右手机保存的原始文件夹按时间配对。
- 对单个设备内部不同模态做时间戳插值和帧率拉齐。
- 输出统一的 seq 目录结构。
- 复制对应 calibration 文件到每个 seq。
- 对旧 subject 自动处理 7 设备数据。
- 对无 STag subject 自动忽略 STag 占位目录。
- 对旧 subject 中同一编号多个 calibration 文件的情况，选择同编号下文件更大的有效 calibration。

典型设备：

```text
Watch_left
Watch_right
Phone_left
Phone_right
Headset
STag_left
STag_right
```

无 STag subject 可能只有：

```text
Watch_left
Watch_right
Phone_left
Phone_right
Headset
```

或特殊情况如 `zhenhong_0529`：

```text
Watch_left
Phone_left
Phone_right
Headset
```

## Step 3: 多设备同步

脚本：

```text
align_sensor_auto.py
```

输入：

```text
data/raw/sensor/<subject>/seq_xx/
```

输出：

```text
data/raw/sensor/<subject>/seq_xx/sensor_data.pt
data/raw/sensor/<subject>/seq_xx/acc_alignment.png
data/raw/manifests/<subject>/align_sensor_auto.json
```

核心逻辑：

- 参与者采集开始时做向上跳跃动作。
- 每个设备在前 `jump_search_frames` 内单独寻找加速度模长最大峰值。
- 默认保留峰值前 `pre_peak_frames=200` 帧。
- 将各设备从各自 `peak-200` 处裁剪，使 jump peak 对齐。
- 所有设备裁到共同最短长度。
- 将传感器数据从约 100fps 下采样到 30fps。
- 使用 calibration 中的 `RMI` / `RSB` 计算：

```text
aM:  global acceleration, [N, D, 3]
RMB: global device/bone rotation, [N, D, 3, 3]
```

`acc_alignment.png` 会把所有设备的加速度模长画在同一张图上，并用 offset 分开，方便人工检查开头 jump peak 是否对齐。

## Step 4: sensor-SMPL 自动对齐

脚本：

```text
auto_align_smpl.py
```

输入：

```text
data/raw/sensor/<subject>/seq_xx/sensor_data.pt
data/raw/smpl/<subject>/smpl_*/smpl_pose_*.pt
data/raw/smpl/<subject>/smpl_*/smpl_tran_*.pt
```

输出：

```text
data/auto_processed/<subject>/<seq_id>.pt
data/alignment_reports/<subject>/<seq_id>_overlay.png
data/alignment_reports/<subject>/<seq_id>_score_curve.png
data/auto_processed/<subject>/manifest.json
```

序列配对：

- `data/raw/sensor/<subject>/seq_01..seq_N`
- `data/raw/smpl/<subject>/smpl_*`
- 两边按文件夹名排序后一一对应。

对齐信号：

- sensor 侧：读取 `sensor_data.pt` 中的 `aM`，取对应设备加速度模长。
- SMPL 侧：用 `articulate.ParametricModel` 做 forward kinematics，合成关节加速度模长。

常规 subject 使用：

```text
left_wrist
right_wrist
head
```

无右手或设备缺失 subject 会使用 subject-specific preset，例如 `zhenhong_0529` 使用：

```text
left_wrist = Watch_left
head       = Headset
```

自动搜索逻辑：

1. 对 sensor / SMPL 信号做 robust normalize。
2. 做 0.25s moving average 平滑。
3. 计算 event signal：`abs(gradient(signal))`。
4. 用 FFT cross-correlation 找 coarse candidate bias。
5. 在 coarse bias 附近逐帧 refine。
6. 对每个 bias 组合评分：

```text
signal_corr
event_corr
peak_score
overlap_ratio
```

默认 `--save-all` 会保存所有序列，置信度和状态只作为人工筛选参考。

## Step 5: pair 数据格式

最终 pair 数据保存在：

```text
data/auto_processed/<subject>/<seq_id>.pt
```

常见字段：

| Key | Shape | 说明 |
| --- | --- | --- |
| `aM` | `[N, D, 3]` | 全局坐标系下加速度 |
| `RMB` | `[N, D, 3, 3]` | 全局坐标系下旋转矩阵 |
| `acc` | `[N, D, 3]` | 原始加速度 |
| `gyro` | `[N, D, 3]` | 陀螺仪 |
| `mag` | `[N, M, 3]` | 磁力计，通常只有 phone/watch |
| `quaternion` | `[N, D, 4]` | 原始四元数 |
| `linear_acc` | `[N, M, 3]` | 线性加速度，通常只有 phone/watch |
| `ppg` | `[N, W, 11]` | 手表 PPG |
| `pose_gt` | `[N, 24, 3, 3]` | 对齐后的 SMPL GT pose |
| `tran_gt` | `[N, 3]` | 对齐后的 SMPL translation |
| `pose_pred` | `[N, 24, 3, 3]` | 模型推理结果，由可视化脚本生成 |
| `frame_bias` | `int` | sensor-SMPL 自动对齐偏移 |

设备顺序通常为：

```text
0 Watch_left
1 Watch_right
2 Phone_left
3 Phone_right
4 Headset
5 STag_left
6 STag_right
```

无 STag subject 的 `D` 可能为 5。`zhenhong_0529` 等缺设备 subject 在 `auto_align_smpl.py` 中会补齐 `aM/RMB` 到 7 设备，并额外保留：

```text
aM_original
RMB_original
device_order
synthetic_device_indices
vi_mask
ji_mask
```

当前合成节点约定：

```python
vi_mask = [1961, 5424, 876, 4362, 411, 3365, 6765]
ji_mask = [18, 19, 1, 2, 15, 7, 8]
```

其中 `aM` 合成来自 `vi_mask` 对应 mesh vertex 的二阶差分加速度，`RMB` 合成来自 `ji_mask` 对应 joint 的全局旋转。

## Step 6: 推理和视频质检

脚本：

```text
visualize_auto_processed.py
```

输入：

```text
data/auto_processed/<subject>/*.pt
data/ckpt/full_model.pth
```

输出：

```text
data/auto_processed/<subject>_videos/<seq_id>.mp4
```

流程：

- 按 `inference.py` 的方式读取 `aM/RMB` 前 5 个 IMU。
- 加载 `data/ckpt/full_model.pth`。
- 每个序列开始前调用 `net.reset()`，避免 online RNN 状态串序列。
- 推理得到 `pose_pred` 并写回 `.pt`。
- 用 SMPL FK 将 `pose_pred` 和 `pose_gt` 转成 24 关节点。
- 离屏渲染 side-by-side 火柴人视频：

```text
left:  pose_pred
right: pose_gt
```

视频里会标注：

```text
seq id
frame index
frame_bias
```

用于人工筛选自动对齐是否合理。

## 主要输出和检查方式

每个 subject 跑完后，应至少检查：

```text
data/raw/manifests/<subject>/full_pipeline.json
data/raw/sensor/<subject>/seq_xx/acc_alignment.png
data/alignment_reports/<subject>/*_overlay.png
data/alignment_reports/<subject>/*_score_curve.png
data/auto_processed/<subject>/*.pt
data/auto_processed/<subject>_videos/*.mp4
```

快速检查数量：

```powershell
(Get-ChildItem data\raw\sensor\<subject> -Directory).Count
(Get-ChildItem data\auto_processed\<subject> -Filter *.pt).Count
(Get-ChildItem data\auto_processed\<subject>_videos -Filter *.mp4).Count
```

快速检查某个 `.pt`：

```powershell
conda run -n baroposer python -c "import torch; d=torch.load('data/auto_processed/<subject>/1.pt',map_location='cpu'); print({k:(tuple(v.shape) if hasattr(v,'shape') else v) for k,v in d.items() if k in ['aM','RMB','pose_gt','pose_pred','tran_gt','frame_bias']})"
```

## 已跑通过的 subject 示例

以下 subject 已用当前流程跑通过：

```text
hyq_0402
yl_0403
lisha_0407
huohuo_0407
chaoran_0529
zhenhong_0529
```

其中：

- `hyq_0402`, `yl_0403`, `lisha_0407`, `huohuo_0407` 是旧 7 设备 subject。
- `chaoran_0529` 无左右 STag，使用 5 设备。
- `zhenhong_0529` 无 STag 且无右手，使用 subject-specific mapping 并在 pair 中补齐 `aM/RMB`。

## 常见问题

### 1. 脚本看起来卡住

通常不是死锁，而是某些步骤耗时较长：

- `process_sensor_auto.py` 对长序列会运行数分钟。
- `visualize_auto_processed.py` 会逐帧推理和渲染，长序列可能需要几十分钟到一个多小时。

可以先跳过视频：

```powershell
conda run -n baroposer python run_subject_pipeline.py --subject <subject> --skip-visualize
```

确认 pair 数据无误后，再单独跑视频：

```powershell
conda run -n baroposer python visualize_auto_processed.py --input-dir data/auto_processed/<subject> --output-dir data/auto_processed/<subject>_videos --overwrite-videos --overwrite-pred
```

### 2. Windows 能否运行 bash

如果 Windows 安装了 WSL / Git Bash，可以运行：

```bash
bash run_subject_pipeline.sh <subject>
```

但更稳定的 Windows 原生命令是：

```powershell
conda run -n baroposer python run_subject_pipeline.py --subject <subject>
```

### 3. 无 STag subject 是否需要手动传 `--no-stag`

通常不需要。`run_subject_pipeline.py` 会读取 calibration 中的 `RMI` 数量自动判断。

仍可手动指定：

```powershell
conda run -n baroposer python run_subject_pipeline.py --subject <subject> --no-stag
```

### 4. 只想重新跑某一步

总控脚本提供 skip 参数：

```powershell
--skip-process-sensor
--skip-align-sensor
--skip-align-smpl
--skip-visualize
```

例如只重跑 sensor-SMPL 对齐和视频：

```powershell
conda run -n baroposer python run_subject_pipeline.py --subject <subject> --skip-process-sensor --skip-align-sensor
```

