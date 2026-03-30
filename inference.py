import articulate as art
import torch
import numpy as np
from utils.model_utils import load_model
from config import paths, amass
from tqdm import tqdm
import os

if __name__ == "__main__":
    '''
    将xingying导出的数据和处理好的其他系统数据放在用同一个路径(imu_path)，
    对齐后得到的数据会保存在相同的路径
    '''
    data_dir = './data/processed'
    sub_name = 'hyq_0327'
    
    sub_dir = os.path.join(data_dir, sub_name)
    
    seq_names = os.listdir(sub_dir)
    seq_num = len(seq_names)
    body_model = art.ParametricModel(paths.smpl_file)

    ckpt_path = 'data/ckpt/full_model.pth'
    net = load_model(ckpt_path)
    net.eval()
    print('Mobileposer model loaded.')
    
    with torch.no_grad():
        for i in range(0, len(seq_names)):
            print(f'Processing sequence {i+1}/{seq_num}...')
            
            seq_name = str(i+1) + ".pt"
            data = torch.load(os.path.join(sub_dir, seq_name))
            
            aM, RMB = data['aM'], data['RMB']
            
            # acc scale
            aM = aM / amass.acc_scale
            aM, RMB = aM[:, :5], RMB[:, :5]
            input = torch.cat((aM.flatten(1), RMB.flatten(1)), dim=1).to("cuda")
            
            poses = []
            for i in tqdm(range(input.shape[0])):
                pose = net.forward_frame(input[i])
                poses.append(pose.cpu())
            
            poses = torch.stack(poses)
            data['pose_pred'] = poses
            torch.save(data, os.path.join(sub_dir, seq_name))
            print(f'Pose prediction for sequence {i+1} saved.')
        
        
