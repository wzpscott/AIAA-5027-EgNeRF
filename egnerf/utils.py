import sys
import typing as t
from dataclasses import dataclass
from rich.progress import Console
from nerfstudio.data.dataparsers.base_dataparser import DataParserConfig
import numpy as np
import torch
import torch.nn as nn
if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points
CONSOLE = Console(width=120)
import tyro


def events_to_image(xs, ys, ps, H, W):
    coords = (xs.astype(int), ys.astype(int))
    abs_coords = np.ravel_multi_index(coords, (H, W))
    img = np.bincount(abs_coords, weights=ps, minlength=H * W)
    img = img.reshape([H, W, 1])
    return img

def descript_events(xs, ys, ps, ts):
    print('shape:', xs.shape, ys.shape, ps.shape, ts.shape)
    print('H: ', xs.max())
    print('W: ', ys.max())
    print('p range', ps.min(), ps.max())
    print('t range', ts.min(), ts.max())
    
def plot_event_frame(event_frame, threhold=3, plot_single_frame=False):
    """
    plot event frame with red (positive events), white (no events) and blue (negative events).
    input:
        event_frame: [N, H, W, C], event frame. 
                    If the event frame is color event data, i.e. C!=1, this function takes average in the last dimension.
        threhold: The threhold for count of events on each pixel for visualization. 
        plot_single_frame: whether to plot a single event frame. 
                            If True, the shape of event_frame will be [H, W, C] and the shape of event_frame_viz will be [H, W, 3]
    output:
        event_frame_viz: [N, H, W, 3]
    """
    event_frame = anything_to_array(event_frame)
    event_frame = event_frame.squeeze(-1)
    if plot_single_frame:
        event_frame = event_frame[None, ...] 
    N, H, W = event_frame.shape
    background = np.ones([N, H, W, 3])
    foreground = np.ones([N, H, W, 3])
    alpha = np.zeros([N, H, W, 1])
    
    foreground[event_frame>0] = np.array([1, 0, 0])
    foreground[event_frame<0] = np.array([0, 0, 1])
    alpha[event_frame>0] = event_frame[event_frame>0, None].clip(0, threhold) / threhold
    alpha[event_frame<0] = -event_frame[event_frame<0, None].clip(-threhold, 0) / threhold
    alpha[np.abs(event_frame)<0.1] = 0
    event_frame_viz = foreground*alpha + background*(1-alpha)
    if plot_single_frame:
        event_frame_viz = event_frame_viz[0]
        
    event_frame_viz = (event_frame_viz*255).astype(np.uint8)
    return event_frame_viz


def anything_to_array(x, dtype=float):
    """
    convert array-like objects(list, torch.tensor, etc.) to np.array 
    """
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy().astype(dtype)
    else:
        x = np.asarray(x, dtype=dtype)
    return x

def stable_minmax_scale(x, q=0.01):
    stable_min = torch.quantile(x, q)
    stable_max = torch.quantile(x, 1-q)
    x = x.clip(stable_min, stable_max)
    x = (x - stable_min) / (stable_max - stable_min)
    return x


# @dataclass
# class DataParserSpecification:
#     """
#     DataParser specification class used to register custom dataparsers with Nerfstudio.
#     The registered dataparsers will be available in commands such as `ns-train`
#     """

#     config: DataParserConfig
#     """Dataparser configuration"""


# def discover_dataparsers() -> t.Dict[str, DataParserConfig]:
#     """
#     Discovers all dataparsers registered using the `nerfstudio.dataparser_configs` entrypoint.
#     """
#     dataparsers = {}
#     discovered_entry_points = entry_points(group="nerfstudio.dataparser_configs")
#     for name in discovered_entry_points.names:
#         spec = discovered_entry_points[name].load()
#         if not isinstance(spec, DataParserSpecification):
#             CONSOLE.print(
#                 f"[bold yellow]Warning: Could not entry point {spec} as it is an instance of DataParserSpecification"
#             )
#             continue
#         spec = t.cast(DataParserSpecification, spec)
#         dataparsers[name] = spec.config
        
#     external_dataparsers = discover_dataparsers()
#     all_dataparsers = {**dataparsers, **external_dataparsers}
#     AnnotatedDataParserUnion = tyro.conf.OmitSubcommandPrefixes[  # Omit prefixes of flags in subcommands.
#         tyro.extras.subcommand_type_from_defaults(
#             all_dataparsers,
#             prefix_names=False,  # Omit prefixes in subcommands themselves.
#         )
#     ]
