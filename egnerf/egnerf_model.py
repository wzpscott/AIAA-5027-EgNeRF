from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.ray_samplers import PDFSampler
from nerfstudio.model_components.renderers import DepthRenderer
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.models.vanilla_nerf import NeRFModel, NeRFModel
from nerfstudio.models.instant_ngp import InstantNGPModelConfig, NGPModel
from nerfstudio.utils.colormaps import apply_colormap
from nerfstudio.viewer.server.viewer_elements import *
from torch.nn import Parameter
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
)
from torchtyping import TensorType
from typing import Optional
from nerfstudio.model_components.renderers import RGBRenderer

@dataclass
class EgNeRFModelConfig(NerfactoModelConfig):
    _target: Type = field(default_factory=lambda: EgNeRFModel)
    rgb_loss_mult: float = 1.0
    event_loss_mult: float = 1.0
    event_threshold: float = 0.25
    
class EgNeRFModel(NerfactoModel):
    config: EgNeRFModelConfig
    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        self.event_loss = MSELoss()
        self.renderer_rgb = HdrRGBRenderer(background_color=self.config.background_color)

    def get_outputs(self, ray_bundle: RayBundle, mode='train'):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        if mode == 'train':
            # hdr_rgb_prev, hdr_rgb = torch.chunk(rgb, 2, dim=0)
            rgb_prev, rgb = torch.chunk(rgb, 2, dim=0)
            depth_prev, depth = torch.chunk(depth, 2, dim=0)
            accumulation_prev, accumulation = torch.chunk(accumulation, 2, dim=0)
            outputs = {
                "rgb": rgb,
                "rgb_prev": rgb_prev,
                "accumulation": accumulation,
                "depth": depth,
            }
        elif mode == 'eval':
            outputs = {
                "rgb": rgb,
                # "rgb_prev": rgb_prev,
                "accumulation": accumulation,
                "depth": depth,
            }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs
    
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        event_frame = batch["event_frame"].to(self.device) * self.config.event_threshold
        color_mask = batch["color_mask"].to(self.device)
    
        eps = 1e-5
        diff = torch.log(outputs["rgb"]**2.2+eps) - torch.log(outputs["rgb_prev"]**2.2+eps)
        # diff = outputs["rgb"] - outputs["rgb_prev"]
        event_frame *= color_mask
        diff *= color_mask
        
        event_loss = ((event_frame - diff)**2).mean()
        loss_dict["event_loss"] = self.config.event_loss_mult * event_loss
        
        bkgd_mask = (event_frame.sum(dim=-1, keepdim=True)==0).float()
        bkgd_color = 159/255
        loss_dict["bkgd_loss"] = ((outputs["rgb"]*bkgd_mask - bkgd_color)**2).mean()
        
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"]) * 0
        
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]

        return loss_dict
    
    def forward(self, ray_bundle: RayBundle, mode='train') -> Dict[str, torch.Tensor]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        return self.get_outputs(ray_bundle, mode=mode)
    
    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward(ray_bundle=ray_bundle, mode='eval')
            for output_name, output in outputs.items():  # type: ignore
                if not torch.is_tensor(output):
                    # TODO: handle lists of tensors as well
                    continue
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs

class HdrRGBRenderer(RGBRenderer):
    """Standard volumetric rendering.

    Args:
        background_color: Background color as RGB. Uses random colors if None.
    """

    def __init__(self, background_color) -> None:
        super().__init__(background_color)

    def forward(
        self,
        rgb: TensorType["bs":..., "num_samples", 3],
        weights: TensorType["bs":..., "num_samples", 1],
        ray_indices: Optional[TensorType["num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> TensorType["bs":..., 3]:
        """Composite samples along ray and render color image

        Args:
            rgb: RGB for each sample
            weights: Weights for each sample
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.

        Returns:
            Outputs of rgb values.
        """

        if not self.training:
            rgb = torch.nan_to_num(rgb)
        rgb = self.combine_rgb(
            rgb, weights, background_color=self.background_color, ray_indices=ray_indices, num_rays=num_rays
        )
        if not self.training:
            torch.clamp_(rgb, min=0.0, max=1.0)
        return  rgb