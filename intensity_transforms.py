import numbers
import random
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union

import numpy as np
import torch

import functional as func
from base import BaseTransform


class ImageOnlyTransform(BaseTransform):
    """
    Transform applying on only image semantic types,
    which are the data having 'image' in the key.
    """

    def apply_function_on_key(self, key: str, data: Any):
        if "image" in key:
            data = self.apply_to_image(data)
        return data


class ContrastStretching(ImageOnlyTransform):
    def __init__(
            self,
            p=0.5,
            always_apply=False,
            gamma=(1.0, 3.0),
            middle_point=(0.3, 0.7),
            rescale=False,
            spatial_dimension_index=-1,
    ):
        super(ContrastStretching, self).__init__(p, always_apply)
        self.rescale = rescale
        self.gamma = self.check_positive_numbers_or_range(gamma, "gamma")
        self.middle_point = self.check_positive_numbers_or_range(middle_point, "middle_point")
        self.spatial_dimension_index = self.check_dimension_index(spatial_dimension_index, "spatial_dimension_index")

    def apply_to_image(self, data: torch.Tensor):
        if self.spatial_dimension_index == -1:
            data = func.contrast_strenching(
                data, gamma=self.params["gamma"], middle_point=self.params["middle_point"], rescale=self.rescale
            )
        else:
            # apply on the given dimension index onwards,
            # typical use-case is to apply different transforms channel-wise.
            data = [
                func.contrast_strenching(
                    s_data, gamma=self.params["gamma"], middle_point=self.params["middle_point"], rescale=self.rescale
                )
                for s_data in torch.split(data, dim=self.spatial_dimension_index, split_size_or_sections=1)
            ]
            data = torch.cat(data, dim=self.spatial_dimension_index)
        return data

    def get_params(self, data_dict: Dict[str, torch.Tensor]):
        gamma = (
            random.uniform(self.gamma[0], self.gamma[1]) if not isinstance(self.gamma, numbers.Number) else self.gamma
        )
        middle_point = (
            random.uniform(self.middle_point[0], self.middle_point[1])
            if not isinstance(self.middle_point, numbers.Number)
            else self.middle_point
        )

        return {
            "gamma": gamma,
            "middle_point": middle_point,
            "spatial_dimension_index": self.spatial_dimension_index,
        }

    def get_transform_init_args_names(self):
        return ("gamma", "middle_point", "rescale", "spatial_dimension_index")


class IntensityWindow(ImageOnlyTransform):
    """
    The combination of intensity rescaling, clipping, and quantization in one class.
    """

    def __init__(
            self, from_span: Tuple[int, int] = (-1100, 400), to_span: Tuple[int, int] = (0, 255),
            output_dtype=torch.uint8
    ):
        super(IntensityWindow, self).__init__(1.0, True, freeze_param=True)
        self.from_span = self.check_range(from_span, "from_span")
        self.to_span = self.check_range(to_span, "to_span")
        self.output_dtype = output_dtype

    def apply_to_image(self, data: torch.Tensor):
        data = func.intensity_window(
            data, from_span=self.from_span, to_span=self.to_span, output_dtype=self.output_dtype
        )
        return data

    def get_transform_init_args_names(self):
        return ("from_span", "to_span", "output_dtype")


class Standardize(ImageOnlyTransform):
    def __init__(self):
        super(Standardize, self).__init__(1.0, True, freeze_param=True)

    def apply_to_image(self, data: torch.Tensor):
        data -= data.mean()
        data /= data.std()
        return data

    def get_transform_init_args_names(self):
        return tuple()


class GaussianSmooth(ImageOnlyTransform):

    def __init__(
            self, p: float = 0.5, always_apply: bool = False, sigma: Tuple[float, float] = (0.5, 2.0),
            truncate: float = 4.0
    ):

        super(GaussianSmooth, self).__init__(p, always_apply)
        self.sigma = self.check_positive_numbers_or_range(sigma, "sigma")
        self.truncate = truncate

    def get_params(self, data_dict: Dict[str, torch.Tensor]):
        sigma = (
            random.uniform(self.sigma[0], self.sigma[1]) if not isinstance(self.sigma, numbers.Number) else self.sigma
        )
        return {
            "sigma": sigma,
            "truncate": self.truncate,
        }

    def get_transform_init_args_names(self):
        return ("sigma", "truncate")

    def apply_to_image(self, data: torch.Tensor):
        data = func.gaussian_smooth(data, sigma=self.params["sigma"], truncate=self.truncate)
        return data


class GaussianAddictive(ImageOnlyTransform):
    def __init__(
            self, p: float = 0.5, always_apply: bool = False, sigma: Tuple[float, float] = (0.03, 0.06)
    ):
        super(GaussianAddictive, self).__init__(p, always_apply)
        self.sigma = self.check_positive_numbers_or_range(sigma, "sigma")

    def get_params(self, data_dict: Dict[str, torch.Tensor]):
        sigma = (
            random.uniform(self.sigma[0], self.sigma[1]) if not isinstance(self.sigma, numbers.Number) else self.sigma
        )
        return {
            "sigma": sigma,
        }

    def get_transform_init_args_names(self):
        return ("sigma")

    def apply_to_image(self, data: torch.Tensor):
        sigma = self.params['sigma']
        d_min = data.min()
        d_max = data.max()
        d_range = d_max - d_min
        # rescale to 0-1
        data_rescale = ((data - d_min) / float(d_range + 1e-7))
        # data_rescale_mean = np.mean(data_rescale)
        noise = sigma * torch.randn(data.shape).type(data.type())
        data_rescale += noise
        data_rescale[data_rescale < 0] = 0.0
        data_rescale[data_rescale > 1] = 1.0
        # rescale back to the original range
        data = data_rescale * d_range + d_min
        return data


class BoxMaskOut(ImageOnlyTransform):
    def __init__(
            self,
            p: float,
            always_apply: bool,
            n_masks: Union[int, Tuple[int, int]],
            region_range: Tuple[float, float] = (0.2, 0.8),
            region_size: Tuple[float, float] = (0.01, 0.06),
            assign_value: int = 0,
            inplace: bool = False,
            freeze_param: bool = False,
    ):
        super(BoxMaskOut, self).__init__(p, always_apply, freeze_param=freeze_param)
        self.region_range = self.check_positive_range(region_range, "region_range")
        self.region_size = self.check_positive_range(region_size, "region_size")
        self.n_masks = self.check_positive_numbers_or_range(n_masks, "n_masks")
        self.assign_value = assign_value
        self.inplace = inplace

    def get_params(self, data_dict: Dict[str, torch.Tensor]):
        sample_img = data_dict["image"]
        ndim = sample_img.dim()
        n_masks = (
            random.randint(self.n_masks[0], self.n_masks[1])
            if isinstance(self.n_masks, (tuple, list))
            else self.n_masks
        )
        mask_centers = [
            tuple([np.random.uniform(self.region_range[0], self.region_range[1]) for _ in range(ndim)])
            for _ in range(n_masks)
        ]
        mask_sizes = [
            tuple([np.random.uniform(self.region_size[0], self.region_size[1]) for _ in range(ndim)])
            for _ in range(n_masks)
        ]
        return {"n_masks": n_masks, "mask_centers": mask_centers, "mask_sizes": mask_sizes}

    def get_transform_init_args_names(self):
        return ("region_range", "region_size", "n_masks", "assign_value", "inplace")

    def apply_to_image(self, data: torch.Tensor):
        mask_centers = self.params["mask_centers"]
        mask_sizes = self.params["mask_sizes"]
        if not self.inplace:
            data = torch.clone(data)

        for mask_center, mask_size in zip(mask_centers, mask_sizes):
            crop_slices = tuple(
                [
                    slice(
                        max(0, int(mc * ds) - int(ms * ds) // 2),
                        min(int(mc * ds) + (int(ms * ds) - int(ms * ds) // 2), ds),
                    )
                    for mc, ds, ms in zip(mask_center, data.shape, mask_size)
                ]
            )
            data[crop_slices] = self.assign_value
        return data
