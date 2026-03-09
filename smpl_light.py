import torch
import numpy as np
# 没smpl模型文件也能运行！用于纯姿态的IK&FK

# SMPL骨架
EDGES = {1:[[0, 1], [0, 2], [0, 3]],
         2:[[1, 4], [2, 5], [3, 6]],
         3:[[4, 7], [5, 8], [6, 9]],
         4:[[7, 10], [8, 11], [9, 12], [9, 13], [9, 14]],
         5:[[12, 15], [13, 16], [14, 17]],
         6:[[16, 18], [17, 19]],
         7:[[18, 20], [19, 21]],
         8:[[20, 22], [21, 23]]}

class SMPLight:
    def __init__(self):
        # 父节点-子节点映射, 用于IK
        self.pc_mapping = []
        # 分层次的父节点-子节点映射, 用于FK
        self.layered_pc_mapping = {}
        for k, v in EDGES.items():
            self.pc_mapping += v
            v = torch.LongTensor(v)
            self.layered_pc_mapping.update({k:[np.array(v[:, 0]).tolist(), np.array(v[:, 1]).tolist()]})
        self.pc_mapping = torch.LongTensor(self.pc_mapping)
        self.pc_mapping = [np.array(self.pc_mapping[:, 0]).tolist(), np.array(self.pc_mapping[:, 1]).tolist()]

    @torch.no_grad()
    def forward_kinematics(self, R):
        for _, mapping in self.layered_pc_mapping.items():
            p_idx, c_idx = mapping[0], mapping[1]
            R[..., c_idx, :, :] = R[..., p_idx, :, :].matmul(R[..., c_idx, :, :])
        return R

    @torch.no_grad()
    def inverse_kinematics(self, R):
        p_idx, c_idx = self.pc_mapping[0], self.pc_mapping[1]
        R[..., c_idx, :, :] = R[..., p_idx, :, :].transpose(-2, -1).matmul(R[..., c_idx, :, :])
        return R