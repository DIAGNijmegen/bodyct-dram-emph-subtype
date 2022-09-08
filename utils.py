import logging
import os, csv
from typing import Type

from pathlib import Path
from typing import Dict

from typing import List

from typing import Union
import SimpleITK as sitk
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers import TensorBoardLogger
import math
from omegaconf import OmegaConf
from pkg_resources import resource_filename
from scipy import ndimage
import hydra
import torch.distributed as dist

logger = logging.getLogger(__name__)


def windowing(image, from_span=(-1150, 350), to_span=(0, 255)):
    if from_span is None:
        min_input = np.min(image)
        max_input = np.max(image)
    else:
        min_input = from_span[0]
        max_input = from_span[1]
    image = np.clip(image, a_min=min_input, a_max=max_input)
    image = ((image - min_input) / float(max_input - min_input)) * (to_span[1] - to_span[0]) + to_span[0]
    return image


def read_csv_in_dict(csv_file_path, column_key, fieldnames=None):
    row_dict = {}
    if not os.path.exists(csv_file_path):
        return row_dict, None
    with open(csv_file_path, "rt") as fp:
        cr = csv.DictReader(fp, delimiter=',', fieldnames=fieldnames)
        for row in cr:
            row_dict[row[column_key]] = row

        field_names = cr.fieldnames
    return row_dict, field_names


def find_crops(mask, spacing, border):
    object_slices = ndimage.find_objects(mask > 0)[0]
    if border > 0:
        pad_object_slices = tuple([
            slice(max(0, os.start - int(math.ceil(border / sp))), min(ss, os.stop + int(math.ceil(border / sp))))
            for os, ss, sp in zip(object_slices, mask.shape, spacing)]
        )
    else:
        pad_object_slices = object_slices

    return pad_object_slices


def cat_all_gather(
        tensors: torch.Tensor
) -> torch.Tensor:
    """
    Performs the concatenated all_reduce operation on the provided tensors.
    """
    gather_sz = dist.get_world_size()
    tensors_gather = [torch.ones_like(tensors) for _ in range(gather_sz)]
    dist.all_gather(
        tensors_gather,
        tensors,
        async_op=False,
    )
    output = torch.cat(tensors_gather, dim=0)
    return output


def get_model_by_name(name):
    config = OmegaConf.load(f"./conf/{name}.yaml")
    return hydra.utils.instantiate(config)

def write_array_to_mha_itk(target_path, arrs, names, type=np.int16,
                           origin=[0.0, 0.0, 0.0],
                           direction=np.eye(3, dtype=np.float64).flatten().tolist(),
                           spacing=[1.0, 1.0, 1.0], orientation='RAI'):
    """ arr is z-y-x, spacing is z-y-x."""
    # size = arrs[0].shape
    for arr, name in zip(arrs, names):
        # assert (arr.shape == size)
        simage = sitk.GetImageFromArray(arr.astype(type))
        simage.SetSpacing(np.asarray(spacing, np.float64).tolist())
        simage.SetDirection(direction)
        simage.SetOrigin(origin)
        fw = sitk.ImageFileWriter()
        fw.SetFileName(target_path + '/{}.mha'.format(name))
        fw.SetDebug(False)
        fw.SetUseCompression(True)
        fw.SetGlobalDefaultDebug(False)
        fw.Execute(simage)


def draw_2d_heatmap(image_2d, masks_2d, alpha=0.5, color_map='jet'):
    blend_image = np.dstack((image_2d, image_2d, image_2d))
    for mask in masks_2d:
        if color_map == 'jet':
            mask_map = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        elif color_map == 'summer':
            mask_map = cv2.applyColorMap(mask, cv2.COLORMAP_SUMMER)
        else:
            raise NotImplementedError
        blend_image = cv2.addWeighted(mask_map, alpha, blend_image, 1 - alpha, 0.0)
    return blend_image


def draw_mask_tile_singleview_heatmap(image, masks_list, coord_mask, num_slices, output_path,
                                      ext='jpg', alpha=0.5, flip_axis=0, draw_anchor=True, zoom_size=360,
                                      anchor_color=(0, 255, 0), colormap='jet', coord_axis=1,
                                      titles=None, title_offset=50, title_color=(0, 255, 0)):
    assert (all([image.shape == mask.shape for mask_list in masks_list for mask in mask_list]))
    if flip_axis is not None:
        image = np.flip(image, axis=flip_axis)
        coord_mask = np.flip(coord_mask, axis=flip_axis)
        m_shape = np.asarray(masks_list).shape
        masks_list = np.asarray([np.flip(mask, axis=flip_axis)
                                 for mask_list in masks_list for mask in mask_list]).reshape(m_shape)

    n_mask_list = len(masks_list)
    n_mask_per_list = len(masks_list[0])
    if zoom_size is not None:
        sp = [image.shape[s] for s in set(list(range(image.ndim))) - {coord_axis}]
        zoom_max_ratio = zoom_size / np.max(sp)
        zoom_ratio = [1.0 if n == coord_axis else zoom_max_ratio for n in range(image.ndim)]

        def zoom_and_pad(i, ratio, target_size, pad_ignore_axis, order):
            i_z = ndimage.zoom(i, ratio, order=order)
            crop_slices = tuple([slice(0, min(n, target_size)) if i != pad_ignore_axis else slice(None, None)
                                 for i, n in enumerate(i_z.shape)])
            i_z = i_z[crop_slices]
            pad_size = tuple([(0, 0) if n == pad_ignore_axis else (
                (target_size - zs) // 2, target_size - zs - (target_size - zs) // 2)
                              for n, zs in zip(range(i.ndim), i_z.shape)])
            i_z_p = np.pad(i_z, pad_size, mode='constant')

            assert (all(i_z_p.shape[n] == target_size for n in range(i.ndim) if n != pad_ignore_axis))
            return i_z_p

        image = zoom_and_pad(image, zoom_ratio, zoom_size, coord_axis, order=1)
        coord_mask = zoom_and_pad(coord_mask, zoom_ratio, zoom_size, coord_axis, order=0)
        masks_list = [zoom_and_pad(mask, zoom_ratio, zoom_size, coord_axis, order=0)
                      for mask_list in masks_list for mask in mask_list]

    if np.sum(coord_mask) > 0:
        foreground_slices = ndimage.find_objects(coord_mask)[0]
        s = foreground_slices[coord_axis].start
        e = foreground_slices[coord_axis].stop
        stride = (e - s) // num_slices
        if stride == 0:
            e = coord_mask.shape[coord_axis] - 1
            s = 0
            stride = (e - s) // num_slices
        slices_ids = list(range(s, e, stride))[:num_slices]
        assert (len(slices_ids) == num_slices)
    else:
        print("no object found!")
        return

    all_slice_tiles = []
    for slice_id in slices_ids:
        # form one slice source from image and masks.
        slice_image = np.take(image, slice_id, axis=coord_axis)
        slice_image_tiles = [np.dstack((slice_image, slice_image, slice_image))]
        for mask_list_id in range(n_mask_list):
            masks = masks_list[mask_list_id * n_mask_per_list: mask_list_id * n_mask_per_list + n_mask_per_list]
            mask_array = [np.take(mask, slice_id, axis=coord_axis) for mask in masks]
            rendered_image = draw_2d_heatmap(slice_image, mask_array, alpha=alpha, color_map=colormap)
            if titles:
                cv2.putText(rendered_image, titles[mask_list_id], (title_offset, title_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, title_color, 1, cv2.LINE_AA)
            slice_image_tiles.append(rendered_image)
        # put all sources into a tile
        slice_image_tiles = np.vstack(slice_image_tiles)
        all_slice_tiles.append(slice_image_tiles)
    draw_ = np.hstack(all_slice_tiles)
    pad_size = ((0, 0), ((1920 - draw_.shape[1]) // 2, (1920 - draw_.shape[1]) - (1920 - draw_.shape[1]) // 2), (0, 0))
    draw_ = np.pad(draw_, pad_size, mode="constant")
    if output_path:

        output_path = Path(output_path).absolute()
        if not os.path.exists(output_path.parent):
            os.makedirs(output_path.parent)
        cv2.imwrite(str(output_path) + '.{}'.format(ext), draw_)


def read_csv_in_dict(csv_file_path, column_key, fieldnames=None):
    row_dict = {}
    if not os.path.exists(csv_file_path):
        return row_dict, None
    with open(csv_file_path, "rt") as fp:
        cr = csv.DictReader(fp, delimiter=',', fieldnames=fieldnames)
        for row in cr:
            row_dict[row[column_key]] = row

        field_names = cr.fieldnames
    return row_dict, field_names


def find_crops(mask, spacing, border):
    object_slices = ndimage.find_objects(mask > 0)[0]
    if border > 0:
        pad_object_slices = tuple([
            slice(max(0, os.start - int(math.ceil(border / sp))), min(ss, os.stop + int(math.ceil(border / sp))))
            for os, ss, sp in zip(object_slices, mask.shape, spacing)]
        )
    else:
        pad_object_slices = object_slices

    return pad_object_slices


def load_state_dict_greedy(model: torch.nn.Module, state_dict_to_load: Dict):
    model_state_dict = model.state_dict()

    for key, weight_to_load in state_dict_to_load.items():
        # saved_key = key.replace('module', 'model')
        saved_key = key
        if saved_key in model_state_dict:
            model_weight = model_state_dict[saved_key]
            shapes_are_matching = model_weight.shape == weight_to_load.shape
            if shapes_are_matching:
                logger.info(f'[load_state_dict_greedy]:correctly loading:{key}')
                model_state_dict[key] = weight_to_load
            else:
                logger.warning(f'[load_state_dict_greedy]:shape mismatch:{key}')
        else:
            logger.warning(f'[load_state_dict_greedy]:unexpected entry:{key}')

    for key in model_state_dict.keys():
        # saved_key = key.replace('model', 'module')
        saved_key = key
        if saved_key not in state_dict_to_load.keys():
            logger.warning(f'[load_state_dict_greedy]:missing entry:{key}')

    model.load_state_dict(model_state_dict, strict=False)


def extract_logger(loggers: List[LightningLoggerBase], logger_class: Type[LightningLoggerBase]):
    if loggers is None:
        return

    for logger in loggers:
        if isinstance(logger, logger_class):
            return logger


def get_loggers(exp_dir: Path):
    loggers = [TensorBoardLogger(save_dir=exp_dir.as_posix(), name="tb_logs")]
    return loggers


def plot_to_numpy_array(plot: plt.Axes) -> np.ndarray:
    canvas = FigureCanvas(plot.get_figure())
    canvas.draw()
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close('all')
    return image


def save_image(image_path: Union[str, Path], rgb_array: np.ndarray) -> None:
    assert rgb_array.dtype in [np.uint8, np.float32, np.float16]

    if rgb_array.dtype == np.float32 or rgb_array.dtype == np.float16:
        rgb_array = np.uint8(rgb_array * 255)

    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(image_path), bgr_array)


def expand_tensor_dims(tensors, expected_dim):
    if tensors.dim() < expected_dim:
        for n in range(expected_dim - tensors.dim()):
            tensors = tensors.unsqueeze(0)

    return tensors


def squeeze_tensor_dims(tensors, expected_dim, squeeze_start_index=0):
    if tensors.dim() > expected_dim:
        for n in range(tensors.dim() - expected_dim):
            tensors = tensors.squeeze(squeeze_start_index)

    return tensors
