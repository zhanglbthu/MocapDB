# 数据集使用说明
## 文件结构
```
Dataset/
├── data/
│   ├── processed/
│   │   └── subject/
│   │       └── x.pt (后处理后的数据)
│   ├── raw/
│   │   ├── sensor/
│   │   │   └── subject/
│   │   │       └── x.pt (原始传感器数据)
│   │   ├── smpl/
│   │   │   └── subject/ (视频动捕结果)
│   │   └── xingying/
│   │       └── subject/ (光学动捕结果)
│   └── overview.csv (数据集概览)
├── align_smpl.py
├── align_xingying.py
├── process_sensor.py
```
## 数据格式
### 后处理后的数据
包含时序同步的传感器数据，SMPL数据以及视频数据，具体包括：
- raw_acc: [N, D, 3]
- acc: [N, D, 3]
- ori: [N, D, 3, 3,]
- gyro: [N, D, 3]
- mag: [N, D, 3]
- pressure: [N, D, 1]
- ppg: [N, D, 1]
- pose_gt: [N, 24, 3, 3] (视频动捕结果)
- tran_gt: [N, 3]
- pose_gt_new: [N, 24, 3, 3] (上半身经过光学动捕refine后的结果)

其中N, D分别代表序列帧数和设备数量，D=7
由于视频数据过大（每个subject约30G），所以保存在云盘上，视频数据见[视频地址](https://cloud.tsinghua.edu.cn/d/5d0182de985d4ee9b111/)
后处理后的传感器数据和SMPL数据为30fps，视频数据为60fps
### 原始传感器数据
包含一个耳机（头部），两个手表（左右手腕），两个手机（左右大腿口袋）和两个S-Tag（左右脚）的原始传感器数据
### 视频动捕结果
包含由EasyMocap处理后得到的SMPL数据
### 光学动捕结果
包含由基于标记点的光学动捕系统得到的原始数据
### 数据集概览
包含每个subject的基本信息和动作类型等元数据

## 后处理脚本
- process_sensor.py: 处理原始传感器数据，进行时序同步
- align_smpl.py: 对齐传感器数据和视频动捕的smpl数据
- align_xingying.py: 得到光学动捕的smpl结果，并进行时序同步和smpl refine