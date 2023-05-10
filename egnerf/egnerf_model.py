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

@dataclass
class EgNeRFModelConfig(NerfactoModelConfig):
    _target: Type = field(default_factory=lambda: EgNeRFModel)
    rgb_loss_mult: float = 1.0
    event_loss_mult: float = 1.0
    event_threshold: float = 0.1
    
class EgNeRFModel(NerfactoModel):
    config: EgNeRFModelConfig
    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        self.event_loss = MSELoss()

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
        event_frame = batch["event_frame"].to(self.device)
        
        event = (outputs["rgb"] - outputs["rgb_prev"]) * self.config.event_threshold
        loss_dict["rgb_loss"] = self.config.rgb_loss_mult * self.rgb_loss(image, outputs["rgb"])
        loss_dict["event_loss"] = self.config.event_loss_mult * self.event_loss(event, event_frame)
        
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]

        return loss_dict
    