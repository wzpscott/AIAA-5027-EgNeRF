"""
EgNeRF configuration file.
"""
from pathlib import Path
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig

from egnerf.egnerf_datamanager import EgNeRFDataManagerConfig, EgNeRFDataParserConfig
from egnerf.egnerf_model import EgNeRFModelConfig

egnerf_method = MethodSpecification(
    config=TrainerConfig(
        method_name="egnerf",
        output_dir= Path("logs"),
        steps_per_eval_batch=500,
        steps_per_eval_image=100,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=EgNeRFDataManagerConfig(
                dataparser=EgNeRFDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=1e-6, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=EgNeRFModelConfig(
                eval_num_rays_per_chunk=1 << 15
            )
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
        },
        vis="tensorboard"
    ),
    description="Base config for EgNeRF",
)


# from nerfstudio.plugins.registry_dataparser import DataParserSpecification
# from my_method.custom_dataparser import CustomDataparserConfig

# MyDataparser = DataParserSpecification(config=CustomDataparserConfig)
# from nerfstudio.plugins.types import 