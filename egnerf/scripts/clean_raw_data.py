from dataclasses import dataclass
import os
import os.path as path

import tyro
import imageio
import numpy as np
from tqdm import tqdm

@dataclass
class Configs:
    '''
    A script to clean the raw data from EventNeRF. 
    This script extracts the synthetic data and remove redundant files from it. 
    '''
    
    raw_data_dir: str = './raw-data'
    '''Directory of raw data'''
    
    data_dir: str = './data'
    '''Directory to output processed data'''
    
if __name__ == '__main__':
    cfgs = tyro.cli(Configs)
    
    os.makedirs(cfgs.data_dir, exist_ok=True)
    os.makedirs(path.join(cfgs.data_dir, 'synthetic'), exist_ok=True)
    
    for scene in os.listdir(path.join(cfgs.raw_data_dir, 'nerf')):
        tqdm.write(f'process scene {scene}')
        os.makedirs(path.join(cfgs.data_dir, 'synthetic', scene), exist_ok=True)
        for split in ['train', 'test', 'validation']:
            os.makedirs(path.join(cfgs.data_dir, 'synthetic', split), exist_ok=True)
            for file in ['events', 'intrinsics', 'pose', 'rgb']:
                src_dir = path.join(cfgs.raw_data_dir, 'nerf', scene, split, file)
                dst_dir = path.join(cfgs.data_dir, 'synthetic', scene, split)
                os.system(f"cp -r {src_dir} {dst_dir}")
                tqdm.write(f'>>>>> copy {src_dir} to {dst_dir}')
                
        