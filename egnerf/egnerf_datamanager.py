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

class InputDataset(BaseInputDataset):
    """Dataset that returns images AND EVENT FRAMES.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        
    def get_data(self, image_idx: int) -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
        """
        image = self.get_image(image_idx)
        data = {"image_idx": image_idx}
        data["image"] = image
        
        event_frame = self.get_event_frame(image_idx)
        data['event_frame'] = event_frame
        if self.has_masks:
            mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
            data["mask"] = get_image_mask_tensor_from_path(filepath=mask_filepath, scale_factor=self.scale_factor)
            assert (
                data["mask"].shape[:2] == data["image"].shape[:2]
            ), f"Mask and image have different shapes. Got {data['mask'].shape[:2]} and {data['image'].shape[:2]}"
        metadata = self.get_metadata(data)
        data.update(metadata)
        return data

    def get_event_frame(self, image_idx: int) -> Dict:
        event_filename = self._dataparser_outputs.metadata['event_filenames'][image_idx]
        event_frame = np.load(event_filename).astype("float32")
        event_frame = torch.from_numpy(event_frame)
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
    config: VanillaDataManagerConfig
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
    
    def create_train_dataset(self) -> InputDataset:
        """Sets up the data loaders for training"""
        return InputDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> InputDataset:
        """Sets up the data loaders for evaluation"""
        return InputDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )
        
    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        batch = self.train_pixel_sampler.sample(image_batch)
        
        ray_indices = batch["indices"]
        ray_indices_prev = batch["indices"].clone()
        ray_indices_prev[0] = torch.max(ray_indices_prev[0], ray_indices_prev[0]-1)
        ray_indices_all = torch.cat([ray_indices_prev, ray_indices], dim=0)
        ray_bundle = self.train_ray_generator(ray_indices_all)
        return ray_bundle, batch

        
