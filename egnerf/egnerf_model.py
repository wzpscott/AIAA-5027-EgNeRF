from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type
from typing_extensions import Literal

import numpy as np
import torch
import torch.nn as nn
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.viewer.server.viewer_elements import *
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
from nerfstudio.utils import colormaps

@dataclass
class EgNeRFModelConfig(NerfactoModelConfig):
    _target: Type = field(default_factory=lambda: EgNeRFModel)
    tonemapper_mode: Literal['fixed-gamma', 'learned-gamma', 'learned-mlp'] = 'learned-mlp'
    use_original_event_nerf: bool = False
    tonemapper_gamma: float = 1.0
    tonemapper_n_layers: int = 3
    tonemapper_d_hidden: int = 128
    rgb_loss_mult: float = 0.1
    event_loss_mult: float = 1.0
    bkgd_loss_mult: float = 0.1
    event_threshold: float = 0.25
    
class EgNeRFModel(NerfactoModel):
    config: EgNeRFModelConfig
    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        self.tonemapper_mode=self.config.tonemapper_mode
        self.gamma = self.config.tonemapper_gamma
        self.d_hidden = self.config.tonemapper_d_hidden
        self.n_layers = self.config.tonemapper_n_layers
        self.tonemapper = Tonemapper(mode=self.tonemapper_mode, n_layers=self.n_layers, d_hidden=self.d_hidden, gamma=self.gamma)
        
        self.mse_loss = lambda a, b : ((a-b)**2).mean() # Origin MSELoss raise weird data type error
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        
    def get_param_groups(self):
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["tonemapper"] = list(self.tonemapper.parameters())
        param_groups["fields"] = list(self.field.parameters())
        
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle, mode='train'):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb_hdr = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        rgb = self.tonemapper(rgb_hdr)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        if mode == 'train':
            rgb_hdr_prev, rgb_hdr = torch.chunk(rgb_hdr, 2, dim=0)
            rgb_prev, rgb = torch.chunk(rgb, 2, dim=0)
            depth_prev, depth = torch.chunk(depth, 2, dim=0)
            accumulation_prev, accumulation = torch.chunk(accumulation, 2, dim=0)
            outputs = {
                "rgb_hdr": rgb_hdr,
                "rgb_hdr_prev": rgb_hdr_prev,
                "rgb": rgb,
                "rgb_prev": rgb_prev,
                "accumulation": accumulation,
                "depth": depth,
            }
            outputs['gamma'] = self.tonemapper.gamma
            
        elif mode == 'eval':
            outputs = {
                "rgb_hdr": rgb_hdr,
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
        event_frame = batch["event_frame"].to(self.device) * self.config.event_threshold
        color_mask = batch["color_mask"].to(self.device)
        bkgd_mask = (event_frame.sum(dim=-1, keepdim=True)==0).float()
        bkgd_color = (159/256) * torch.ones_like(outputs["rgb_hdr"]) * bkgd_mask
        eps = 1e-5
        
        if self.config.use_original_event_nerf:
            # Origin eventnerf implementation
            diff = torch.log(outputs["rgb_hdr"]**2.2+eps) - torch.log(outputs["rgb_hdr_prev"]**2.2+eps)
            event_frame *= color_mask
            diff *= color_mask
            loss_dict["event_loss"] = self.mse_loss(event_frame, diff)
            loss_dict["bkgd_loss"] = self.mse_loss(outputs["rgb_hdr"]*bkgd_mask, bkgd_color)
            loss_dict["rgb_loss"] = 0 * self.mse_loss(image, outputs["rgb"])
        else:
            diff = torch.log(outputs["rgb_hdr"]+eps) - torch.log(outputs["rgb_hdr_prev"]+eps)
            event_frame *= color_mask
            diff *= color_mask
            loss_dict["event_loss"] = self.config.event_loss_mult * self.mse_loss(event_frame, diff)
            loss_dict["bkgd_loss"] = self.config.bkgd_loss_mult * self.mse_loss(outputs["rgb_hdr"]*bkgd_mask, bkgd_color)
            loss_dict["rgb_loss"] = self.config.rgb_loss_mult * self.mse_loss(image, outputs["rgb"])
        
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
    
    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
            metrics_dict["gamma"] = outputs["gamma"]
        return metrics_dict
    
    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        
        image = batch["image"].to(self.device)
        rgb_hdr = outputs["rgb_hdr"]
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"] / outputs["accumulation"].max())
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )
        images_dict = {"0-GT": image, "1-RGB": rgb, "2-RGB-HDR": rgb_hdr, "3-ACC": acc, "4-DEPTH": depth}
        
        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]
        rgb_hdr = torch.moveaxis(rgb_hdr, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)
        
        psnr_hdr = self.psnr(image, rgb_hdr)
        ssim_hdr = self.ssim(image, rgb_hdr)
        lpips_hdr = self.lpips(image, rgb_hdr)

        # all of these metrics will be logged as scalars
        metrics_dict = {"1.0-psnr": float(psnr.item()), "1.1-ssim": float(ssim)}
        metrics_dict["1.2-lpips"] = float(lpips)
        
        metrics_dict["2.0-psnr-hdr"] = float(psnr_hdr)
        metrics_dict["2.1-ssim-hdr"] = float(ssim_hdr)
        metrics_dict["2.2-lpips-hdr"] = float(lpips_hdr)
        
        if self.config.use_original_event_nerf or self.config.rgb_loss_mult != 0:
            metrics_dict["0.0-report-psnr"] = float(psnr_hdr)
            metrics_dict["0.1-report-ssim"] = float(ssim_hdr)
            metrics_dict["0.2-report-lpips"] = float(lpips_hdr)
        else:
            metrics_dict["0.0-report-psnr"] = float(psnr)
            metrics_dict["0.1-report-ssim"] = float(ssim)
            metrics_dict["0.2-report-lpips"] = float(lpips)

        return metrics_dict, images_dict
    
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


class Tonemapper(nn.Module):
    def __init__(self, mode, n_layers=None, d_hidden=None, gamma=None):
        super().__init__()
        # mode = mode[0]
        assert mode in ['fixed-gamma', 'learned-gamma', 'learned-mlp'], f'No such mode: {mode}'
        self.learnable = ('learned' in mode)
        self.mode = mode
        
        if mode in ['fixed-gamma', 'learned-gamma']:
            assert gamma is not None
            requires_grad = ('learned' in mode)
            self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=requires_grad)
        else: # mlp tonemapper
            assert (n_layers is not None) and (d_hidden is not None)
            self.gamma = nn.Parameter(torch.Tensor([gamma])) # place holder
            self.n_layers = n_layers
            self.d_hidden = d_hidden
            self.d_input = 3
            self.d_output = 3
            
            self.mlp = nn.ModuleList()
            for i_layer in range(n_layers):
                if i_layer == 0:
                    self.mlp.append(nn.Linear(self.d_input, self.d_hidden))
                    self.mlp.append(nn.ReLU())
                elif i_layer < n_layers-1:
                    self.mlp.append(nn.Linear(self.d_hidden, self.d_hidden))
                    self.mlp.append(nn.ReLU())
                else: # last layer
                    self.mlp.append(nn.Linear(self.d_hidden, self.d_output))
                    self.mlp.append(nn.Sigmoid())
            self.mlp = nn.Sequential(*self.mlp)
                    
    def forward(self, rgb_hdr):
        if self.mode in ['fixed-gamma', 'learned-gamma']:
            rgb = (rgb_hdr/(rgb_hdr+1)) ** self.gamma
        else:
            rgb = self.mlp(rgb_hdr)
            
        return rgb
        
        
        

