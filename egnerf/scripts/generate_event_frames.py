from dataclasses import dataclass
import os
import os.path as path
import sys
sys.path.append('/home/zipengwang/workspace/1-AIAA5027-G5/0-EgNeRF/egnerf')

import tyro
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm, trange

from utils import *

@dataclass
class Configs:
    '''
    A script to stack raw event data to event frames.
    '''
    
    data_dir: str = './data/synthetic'
    '''Directory which contains raw event data'''
    
    H: int = 260
    W: int = 346
    viz_threshold: int = 1
    
if __name__ == '__main__':
    cfgs = tyro.cli(Configs)
    
    for scene in os.listdir(cfgs.data_dir):
        tqdm.write(f'process scene {scene}')
        for split in ['train']:
        # for split in ['train']:
            event_frame_dir = path.join(cfgs.data_dir, scene, split, f'event-frame')
            event_frame_viz_dir = path.join(cfgs.data_dir, scene, split, f'event-frame-viz')
            if path.exists(event_frame_dir):
                os.system(f'rm -rf {event_frame_dir}')
            if path.exists(event_frame_viz_dir):
                os.system(f'rm -rf {event_frame_viz_dir}')
            os.makedirs(event_frame_dir)
            os.makedirs(event_frame_viz_dir)
            
            events = np.load(path.join(cfgs.data_dir, scene, split, 'events.npz'))
            xs, ys, ps, ts = events['x'], events['y'], events['p'], events['t']
            xs, ys = ys, xs # e-sim stores x and y in reverse order
            for i in trange(0, 1001):
                if i == 0:
                    event_frame = np.zeros([cfgs.H, cfgs.W, 1])
                else:
                    event_frame = events_to_image(xs[ts==i], ys[ts==i], ps[ts==i], cfgs.H, cfgs.W)
                event_frame_viz = plot_event_frame(event_frame, threhold=cfgs.viz_threshold, plot_single_frame=True)
                
                np.save(path.join(event_frame_dir, f'r_{i:05d}.npy'), event_frame)
                imageio.imwrite(path.join(event_frame_viz_dir, f'r_{i:05d}.png'), event_frame_viz)