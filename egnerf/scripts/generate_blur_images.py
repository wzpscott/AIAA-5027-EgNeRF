from dataclasses import dataclass
import os
import os.path as path

import tyro
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

@dataclass
class Configs:
    '''
    A script to generate images with motion blur from a series of clear images by average consecutive images.
    '''
    
    data_dir: str = './data/synthetic'
    '''Directory which contains input clear images'''
    
    n_images: int = 20
    '''Number of images to generate one blurred image'''

def generate_blur_images(input_dir, output_dir, n_images):
    input_img_fnames = sorted(os.listdir(input_dir))
    
    imgs = []
    for i, img_fname in enumerate(tqdm(input_img_fnames)):
        img = imageio.imread(path.join(input_dir, img_fname))
        imgs.append(img)
        
        if (i+1) % n_images == 0:
            blur_img = np.stack(imgs, axis=0).mean(axis=0).astype(np.uint8)
            imageio.imsave(path.join(output_dir, img_fname), blur_img)
            imgs = []
            # tqdm.write(f'Write blur image to {path.join(output_dir, img_fname)}')
            
if __name__ == '__main__':
    cfgs = tyro.cli(Configs)
    
    for scene in os.listdir(cfgs.data_dir):
        tqdm.write(f'process scene {scene}')
        # for split in ['train', 'test', 'validation']:
        for split in ['train']:
            tqdm.write(f'>>>>> process split {split}')
            input_dir = path.join(cfgs.data_dir, scene, split, 'rgb')
            output_dir = path.join(cfgs.data_dir, scene, split, f'blur-rgb-{cfgs.n_images}')
            if path.exists(output_dir):
                os.system(f'rm -rf {output_dir}')
            os.makedirs(output_dir)
            generate_blur_images(input_dir, output_dir, cfgs.n_images)
    
    