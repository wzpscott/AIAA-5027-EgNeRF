from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Type
import json
from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch
from torchtyping import TensorType
from rich.console import Console
from typing_extensions import Literal

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs
)
from nerfstudio.data.scene_box import SceneBox

CONSOLE = Console(width=120)

def _find_files(directory: str, exts: List[str]):
    """Find all files in a directory that have a certain file extension.

    Args:
        directory : The directory to search for files.
        exts :  A list of file extensions to search for. Each file extension should be in the form '*.ext'.

    Returns:
        A list of file paths for all the files that were found. The list is sorted alphabetically.
    """
    if os.path.isdir(directory):
        # types should be ['*.png', '*.jpg', '*.JPG', '*.PNG']
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(directory, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(files_grabbed)
        return files_grabbed
    return []

def _parse_osm_txt(filename: str):
    """Parse a text file containing numbers and return a 4x4 numpy array of float32 values.

    Args:
        filename : a file containing numbers in a 4x4 matrix.

    Returns:
        A numpy array of shape [4, 4] containing the numbers from the file.
    """
    assert os.path.isfile(filename)
    with open(filename, encoding="UTF-8") as f:
        nums = f.read().split()
    return np.array([float(x) for x in nums]).reshape([4, 4]).astype(np.float32)

def get_camera_params(
    scene_dir: str, split: Literal["train", "validation", "test"]
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Load camera intrinsic and extrinsic parameters for a given scene split.

    Args"
      scene_dir : The directory containing the scene data.
      split : The split for which to load the camera parameters.

    Returns
        A tuple containing the intrinsic parameters (as a torch.Tensor of shape [N, 4, 4]),
        the camera-to-world matrices (as a torch.Tensor of shape [N, 4, 4]), and the number of cameras (N).
    """
    split_dir = f"{scene_dir}/{split}"

    # camera parameters files
    intrinsics_files = _find_files(f"{split_dir}/intrinsics", exts=["*.txt"])
    pose_files = _find_files(f"{split_dir}/pose", exts=["*.txt"])

    num_cams = len(pose_files)

    intrinsics = []
    camera_to_worlds = []
    for i in range(num_cams):
        intrinsics.append(_parse_osm_txt(intrinsics_files[i]))

        pose = _parse_osm_txt(pose_files[i])

        # convert from COLMAP/OpenCV to nerfstudio camera (OpenGL/Blender)
        pose[0:3, 1:3] *= -1

        camera_to_worlds.append(pose)

    intrinsics = torch.from_numpy(np.stack(intrinsics).astype(np.float32))  # [N, 4, 4]
    camera_to_worlds = torch.from_numpy(np.stack(camera_to_worlds).astype(np.float32))  # [N, 4, 4]

    return intrinsics, camera_to_worlds, num_cams

# @dataclass
# class DataparserOutputs:
#     """Dataparser outputs for the which will be used by the DataManager
#     for creating RayBundle and RayGT objects."""

#     image_filenames: List[Path]
#     """Filenames for the images."""
#     event_filenames: List[Path]
#     """Filenames for the event frames."""
#     cameras: Cameras
#     """Camera object storing collection of camera information in dataset."""
#     alpha_color: Optional[TensorType[3]] = None
#     """Color of dataset background."""
#     scene_box: SceneBox = SceneBox()
#     """Scene box of dataset. Used to bound the scene or provide the scene scale depending on model."""
#     mask_filenames: Optional[List[Path]] = None
#     """Filenames for any masks that are required"""
#     metadata: Dict[str, Any] = to_immutable_dict({})
#     """Dictionary of any metadata that be required for the given experiment.
#     Will be processed by the InputDataset to create any additional tensors that may be required.
#     """
#     dataparser_transform: TensorType[3, 4] = torch.eye(4)[:3, :]
#     """Transform applied by the dataparser."""
#     dataparser_scale: float = 1.0
#     """Scale applied by the dataparser."""

#     def as_dict(self) -> dict:
#         """Returns the dataclass as a dictionary."""
#         return vars(self)

#     def save_dataparser_transform(self, path: Path):
#         """Save dataparser transform to json file. Some dataparsers will apply a transform to the poses,
#         this method allows the transform to be saved so that it can be used in other applications.

#         Args:
#             path: path to save transform to
#         """
#         data = {
#             "transform": self.dataparser_transform.tolist(),
#             "scale": float(self.dataparser_scale),
#         }
#         if not path.parent.exists():
#             path.parent.mkdir(parents=True)
#         with open(path, "w", encoding="UTF-8") as file:
#             json.dump(data, file, indent=4)

#     def transform_poses_to_original_space(
#         self,
#         poses: TensorType["num_poses", 3, 4],
#         camera_convention: Literal["opengl", "opencv"] = "opencv",
#     ) -> TensorType["num_poses", 3, 4]:
#         """
#         Transforms the poses in the transformed space back to the original world coordinate system.
#         Args:
#             poses: Poses in the transformed space
#             camera_convention: Camera system convention used for the transformed poses
#         Returns:
#             Original poses
#         """
#         return transform_poses_to_original_space(
#             poses,
#             self.dataparser_transform,
#             self.dataparser_scale,
#             camera_convention=camera_convention,
#         )

@dataclass
class EgNeRFDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: EgNeRF)
    """target class to instantiate"""
    data: Path = Path("./egnerf/data/synthetic")
    """Directory specifying location of data."""
    scene: str = "lego"
    """Which scene to load"""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    use_masks: bool = False
    """Whether to use masks."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "vertical"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "focus"
    """The method to use for centering."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    
    
@dataclass
class EgNeRF(DataParser):
    """EgNeRF Dataparser
    Follow the convention of NeRFOSR, add event data

    Some of this code comes from https://github.com/r00tman/NeRF-OSR/blob/main/data_loader_split.py

    Source data convention is:
      camera coordinate system: x-->right, y-->down, z-->scene (opencv/colmap convention)
      poses is camera-to-world
      masks are 0 for dynamic content, 255 for static content
    """
    config: EgNeRFDataParserConfig
    
    def _generate_dataparser_outputs(self, split="train"):
        data = self.config.data
        scene = self.config.scene
        split = "validation" if split == "val" else split
        
        scene_dir = f"{data}/{scene}"
        split_dir = f"{data}/{scene}/{split}"
        
        # get all split cam params
        intrinsics_train, camera_to_worlds_train, n_train = get_camera_params(scene_dir, "train")
        intrinsics_val, camera_to_worlds_val, n_val = get_camera_params(scene_dir, "validation")
        intrinsics_test, camera_to_worlds_test, _ = get_camera_params(scene_dir, "test")
        
        # combine all cam params
        intrinsics = torch.cat([intrinsics_train, intrinsics_val, intrinsics_test], dim=0)
        camera_to_worlds = torch.cat([camera_to_worlds_train, camera_to_worlds_val, camera_to_worlds_test], dim=0)

        camera_to_worlds, _ = camera_utils.auto_orient_and_center_poses(
            camera_to_worlds,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )
        
        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= torch.max(torch.abs(camera_to_worlds[:, :3, 3]))

        camera_to_worlds[:, :3, 3] *= scale_factor * self.config.scale_factor

        if split == "train":
            camera_to_worlds = camera_to_worlds[:n_train]
            intrinsics = intrinsics[:n_train]
        elif split == "validation":
            camera_to_worlds = camera_to_worlds[n_train : n_train + n_val]
            intrinsics = intrinsics[n_train : n_train + n_val]
        elif split == "test":
            camera_to_worlds = camera_to_worlds[n_train + n_val :]
            intrinsics = intrinsics[n_train + n_val :]

        cameras = Cameras(
            camera_to_worlds=camera_to_worlds[:, :3, :4],
            fx=intrinsics[:, 0, 0],
            fy=intrinsics[:, 1, 1],
            cx=intrinsics[:, 0, 2],
            cy=intrinsics[:, 1, 2],
            camera_type=CameraType.PERSPECTIVE,
        )
        
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )
        
        # --- images ---
        image_filenames = _find_files(f"{split_dir}/rgb", exts=["*.png", "*.jpg", "*.JPG", "*.PNG"])
        
        # --- event frames ---
        event_filenames = _find_files(f"{split_dir}/event-frame", exts=["*.npy"])
        
        # --- masks ---
        mask_filenames = []
        if self.config.use_masks:
            mask_filenames = _find_files(f"{split_dir}/mask", exts=["*.png", "*.jpg", "*.JPG", "*.PNG"])
        
        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            # event_filenames=event_filenames,
            metadata={'event_filenames': event_filenames},
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=self.config.scale_factor,
        )
        return dataparser_outputs