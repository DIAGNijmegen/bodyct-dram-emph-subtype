from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch

import functional as func
from base import BaseTransform
from utils import expand_tensor_dims
from utils import squeeze_tensor_dims
import random


class DualTransform(BaseTransform):

    def apply_function_on_key(self, key: str, data: Any):
        if "image" in key:
            data = self.apply_to_image(data)
        elif "mask" in key:
            data = self.apply_to_mask(data)
        return data

    def apply_to_image(self, data: Any) -> torch.Tensor:
        raise NotImplementedError(f"Method apply_to_image is not implemented in class  {self.__class__.__name__}")

    def apply_to_mask(self, data: Any) -> torch.Tensor:
        raise NotImplementedError(f"Method apply_to_mask is not implemented in class  {self.__class__.__name__}")


class Interpolate(DualTransform):

    def __init__(
        self,
        target_size: Optional[Union[int, Tuple[int, ...]]],
        scale_factor: Optional[Union[float, Tuple[float, ...]]],
        align_corners=False,
        mode=None,
        only_in_plane=True,
    ):
        super(Interpolate, self).__init__(p=1.0, always_apply=True, freeze_param=True)
        self.target_size = self.check_numbers_or_tuple(target_size, "target_size") if target_size else None
        self.scale_factor = self.check_numbers_or_tuple(scale_factor, "scale_factor") if scale_factor else None
        self.align_corners = align_corners
        self.only_in_plane = only_in_plane
        self.mode = mode
        if self.target_size is None and self.scale_factor is None:
            raise ValueError("Either target_size or rescale_factor must be given.")

    def get_transform_init_args_names(self):
        return ("target_size", "scale_factor", "align_corners", "mode", "only_in_plane")

    def apply_to_image(self, data: torch.Tensor):
        ndim = data.dim()
        dtype = data.type()
        assert ndim == 3
        if self.only_in_plane:
            original_d = data.shape[0]
            new_d = self.target_size[0]
            data = expand_tensor_dims(data, 4).float()
            data = torch.nn.functional.interpolate(
                data, size=self.target_size[1:], scale_factor=self.scale_factor, mode='bilinear', align_corners=self.align_corners
            )
            indices = torch.linspace(0, original_d - 1, new_d).long()
            data = data[:, indices, ::]
            data = squeeze_tensor_dims(data, ndim).type(dtype)
        else:
            data = expand_tensor_dims(data, 5).float()
            data = torch.nn.functional.interpolate(
                data, size=self.target_size, scale_factor=self.scale_factor, mode='trilinear', align_corners=self.align_corners
            )
            data = squeeze_tensor_dims(data, ndim).type(dtype)
        return data

    def apply_to_mask(self, data: Any) -> torch.Tensor:
        ndim = data.dim()
        dtype = data.type()
        assert ndim == 3
        if self.only_in_plane:
            original_d = data.shape[0]
            new_d = self.target_size[0]
            data = expand_tensor_dims(data, 4).float()
            data = torch.nn.functional.interpolate(
                data, size=self.target_size[1:], scale_factor=self.scale_factor, mode='nearest'
            )
            indices = torch.linspace(0, original_d - 1, new_d).long()
            data = data[:, indices, ::]
            data = squeeze_tensor_dims(data, ndim).type(dtype)
        else:
            data = expand_tensor_dims(data, 5).float()
            data = torch.nn.functional.interpolate(
                data, size=self.target_size, scale_factor=self.scale_factor, mode='nearest',
            )
            data = squeeze_tensor_dims(data, ndim).type(dtype)
        return squeeze_tensor_dims(data, ndim).type(dtype)


class Flip(DualTransform):

    def __init__(
        self,
        p, always_apply,
        dim: Union[int, Tuple[int, ...]],
    ):
        super(Flip, self).__init__(p=p, always_apply=always_apply)
        self.dim = self.check_positive_numbers_or_range(dim, "dim")

    def get_transform_init_args_names(self):
        return ("dim",)

    def get_params(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        dim_int = np.random.randint(self.dim[0], self.dim[1])
        dim = data_dict['image'].dim()
        combs = random.sample(range(dim), dim_int)
        return {
            "combs": combs
        }

    def apply(self, data: torch.Tensor):
        combs = self.params['combs']

        data = torch.flip(data, dims=combs)
        return data

    def apply_to_image(self, data: torch.Tensor):
        return self.apply(data)

    def apply_to_mask(self, data: Any) -> torch.Tensor:
        return self.apply(data)

class CropAndResize(DualTransform):
    def __init__(
        self,
        p,
        always_apply,
        crop_center: Tuple[float, ...],
        crop_size: Tuple[float, ...],
        position_given=False,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: Optional[bool] = None,
    ):
        super(CropAndResize, self).__init__(p, always_apply)
        self.crop_center = crop_center
        self.crop_size = crop_size
        self.position_given = position_given
        self.align_corners = align_corners
        self.padding_mode = padding_mode
        self.mode = mode

    def get_params(self, data_dict: Dict[str, torch.Tensor]):
        sample_img = data_dict["image"]
        ndim = sample_img.dim()
        if not self.position_given:
            crop_center = tuple([np.random.uniform(self.crop_center[0], self.crop_center[1]) for _ in range(ndim)])
            crop_size = tuple([np.random.uniform(self.crop_size[0], self.crop_size[1]) for _ in range(ndim)])
        else:
            crop_center = self.crop_center
            crop_size = self.crop_size
        return {"crop_center": crop_center, "crop_size": crop_size}

    def get_transform_init_args_names(self):
        return ("crop_center", "crop_size", "position_given", "align_corners", "padding_mode", "mode")

    def apply(self, data: torch.Tensor, **kwargs):
        crop_center = self.params["crop_center"]
        crop_size = self.params["crop_size"]
        bounding_box = np.asarray(
            [
                (max(0, int(mc * ds) - int(ms * ds) // 2), min(int(mc * ds) + (int(ms * ds) - int(ms * ds) // 2), ds))
                for mc, ds, ms in zip(crop_center, data.shape, crop_size)
            ]
        )
        spatial_size = data.shape
        ndim = data.dim()
        dtype = data.type()
        bounding_box = torch.as_tensor(bounding_box).unsqueeze(0) / torch.as_tensor(spatial_size).unsqueeze(
            0
        ).unsqueeze(-1)
        data = func.roi_align(
            expand_tensor_dims(data, ndim + 2).float(),
            bounding_box,
            (1, 1) + tuple(spatial_size),
            mode=kwargs['mode'],
            padding_mode=kwargs.get('padding', self.padding_mode),
            align_corners=kwargs.get('align_corners', self.align_corners),
        )
        data = squeeze_tensor_dims(data, ndim).type(dtype)
        return data

    def apply_to_image(self, data: torch.Tensor):
        return self.apply(data, mode=self.mode)

    def apply_to_mask(self, data: torch.Tensor) -> torch.Tensor:
        return self.apply(data, mode="nearest", padding='zeros', align_corners=False)
