from __future__ import annotations

import os.path as osp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import torch
import numpy as np
import yaml
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.misc import IterableWrapper
from rich.progress import Console

CONSOLE = Console(width=120)

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.data.datasets.base_dataset import InputDataset as BaseInputDataset
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from egnerf.egnerf_dataparser import EgNeRFDataParserConfig
from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path
from nerfstudio.data.pixel_samplers import (
    EquirectangularPixelSampler,
    PatchPixelSampler,
    PixelSampler,
)
from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
)

class InputDataset(BaseInputDataset):
    """Dataset that returns images AND EVENT FRAMES.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0, split='train'):
        super().__init__(dataparser_outputs, scale_factor)
        self.split = split
        self.H, self.W = 260, 346
        self.color_mask = torch.zeros(self.H, self.W, 3)
        self.color_mask[0::2, 0::2, 0] = 1  # r
        self.color_mask[0::2, 1::2, 1] = 1  # g
        self.color_mask[1::2, 0::2, 1] = 1  # g
        self.color_mask[1::2, 1::2, 2] = 1  # b
        
    def get_data(self, image_idx: int) -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
        """
        image = self.get_image(image_idx)
        data = {"image_idx": image_idx}
        data["image"] = image
        
        if self.split == 'train':
            event_frame = self.get_event_frame(image_idx)
            data['event_frame'] = event_frame
        else:
            data['event_frame'] = torch.zeros_like(image)
        data['color_mask'] = self.color_mask
        data['img_idx'] = image_idx * torch.ones_like(image)
        
        if self.split == 'train':
            event_mask = (data['event_frame'].sum(dim=-1)!=0).float()
            neg_mask = (torch.rand_like(event_mask) < 0.1).float()
            mask = ((event_mask + neg_mask) != 0).float()
            mask = mask.unsqueeze(-1).repeat_interleave(3, dim=-1)
            data['mask'] = mask  
            
        metadata = self.get_metadata(data)
        data.update(metadata)
        return data

    def get_event_frame(self, image_idx: int) -> Dict:
        image_idx_prev = max(0, image_idx-50)
        event_filenames = [self._dataparser_outputs.metadata['event_filenames'][i] for i in range(image_idx_prev, image_idx)]
        event_frame = np.zeros([self.H, self.W, 1])
        for event_filename in event_filenames:
            event_frame += np.load(event_filename).astype("float32")
            
        event_frame = torch.from_numpy(event_frame)
        event_frame = torch.repeat_interleave(event_frame, 3, dim=-1)
        return event_frame

@dataclass
class EgNeRFDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: EgNeRFDataManager)
    dataparser = EgNeRFDataParserConfig


class EgNeRFDataManager(VanillaDataManager):  # pylint: disable=abstract-method
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: EgNeRFDataManager
    train_dataset: InputDataset
    eval_dataset: InputDataset
    train_dataparser_outputs: DataparserOutputs

    def __init__(
        self,
        config: EgNeRFDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )
        
    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        assert self.eval_dataset is not None
        CONSOLE.print("Setting up evaluation dataset...")
        self.eval_image_dataloader = CacheDataloader(
            self.eval_dataset,
            num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)
        self.eval_pixel_sampler = self._get_pixel_sampler(self.eval_dataset, self.config.eval_num_rays_per_batch)
        self.eval_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.eval_dataset.cameras.size, device=self.device
        )
        self.eval_ray_generator = RayGenerator(
            self.eval_dataset.cameras.to(self.device),
            self.eval_camera_optimizer,
        )
        # for loading full images
        self.fixed_indices_eval_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )
        self.eval_dataloader = RandIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )
    
    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size, device=self.device
        )
        self.train_ray_generator = RayGenerator(
            self.train_dataset.cameras.to(self.device),
            self.train_camera_optimizer,
        )
        
    def create_train_dataset(self) -> InputDataset:
        """Sets up the data loaders for training"""
        return InputDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> InputDataset:
        """Sets up the data loaders for evaluation"""
        return InputDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
            split='val'
        )
        
    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        batch = self.train_pixel_sampler.sample(image_batch)
        
        ray_indices = batch["indices"]
        ray_indices_prev = batch["indices"].clone()
        ray_indices_prev[:, 0][ray_indices_prev[:, 0]<50] = 50
        ray_indices_prev[:, 0] -= 50
        ray_indices_all = torch.cat([ray_indices_prev, ray_indices], dim=0)
        ray_bundle = self.train_ray_generator(ray_indices_all)
        batch["indices"] = ray_indices_all
        
        return ray_bundle, batch
    
    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)
        assert self.eval_pixel_sampler is not None
        batch = self.eval_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)
        ray_indices_prev = batch["indices"].clone()
        ray_indices_prev[0] = torch.max(ray_indices_prev[0], ray_indices_prev[0]-1)
        ray_indices_all = torch.cat([ray_indices_prev, ray_indices], dim=0)
        ray_bundle = self.eval_ray_generator(ray_indices_all)
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        for camera_ray_bundle, batch in self.eval_dataloader:
            assert camera_ray_bundle.camera_indices is not None
            image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])
            return image_idx, camera_ray_bundle, batch
        raise ValueError("No more eval images")
    

        
