import warnings
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch

epsilon = 1e-7


def intensity_window(img: torch.Tensor, from_span=(-1150, 350), to_span=(0, 255), output_dtype=None) -> torch.Tensor:
    origin_dtype = img.type()
    img = img.float()
    if from_span is None:
        min_input = img.min()
        max_input = img.max()
    else:
        min_input = from_span[0]
        max_input = from_span[1]
    img = torch.clamp(img, min=min_input, max=max_input)
    img = ((img - min_input) / (max_input - min_input)) * (to_span[1] - to_span[0]) + to_span[0]
    if output_dtype is not None:
        if output_dtype != origin_dtype:
            warnings.warn(f"intensity_rescale converting {origin_dtype} to {output_dtype}. ")
        img = img.type(output_dtype)
    return img


def contrast_strenching(
    img: Union[np.ndarray, torch.Tensor], rescale: bool, middle_point: float, gamma: float
) -> Union[np.ndarray, torch.Tensor]:

    d_min = img.min()
    d_max = img.max()
    d_range = d_max - d_min
    if rescale:
        standard_img = (img - d_min) / (d_range + epsilon)
    else:
        assert img.min() >= 0.0 and img.max() <= 1.0, "image should be standardized, not in range [0, 1]."
        standard_img = img
    return 1.0 / (1.0 + ((middle_point / (standard_img + epsilon)) ** gamma))


def generate_gaussian_1d_kernel(sigma: float, truncate: float) -> torch.Tensor:
    assert truncate % 2 == 0, "truncate should be an even number."
    radius = int(truncate * sigma + 0.5)
    sigma2 = sigma**2
    x = torch.arange(-radius, radius + 1)
    phi_x = torch.exp(-0.5 / sigma2 * x**2)
    phi_x = phi_x / phi_x.sum()
    return phi_x


@torch.no_grad()
def gaussian_smooth(img: torch.Tensor, sigma: float, truncate: float = 4.0) -> torch.Tensor:
    kernel = generate_gaussian_1d_kernel(sigma, truncate)
    kernel = kernel.to(img.device)
    k1d = kernel.view(1, 1, -1)
    origin_shape = img.shape
    for _ in range(img.dim()):
        img = torch.nn.functional.conv1d(img.view(-1, 1, img.shape[-1]), k1d, padding="same").view(*img.shape)
        img = img.permute(2, 0, 1).contiguous()
    assert origin_shape == img.shape, "gaussian smooth should reserve the input shape (using same padding)."
    return img


def compute_crop_resize_affine_matrix(bounding_boxes: torch.Tensor) -> torch.Tensor:
    ndim = bounding_boxes.shape[1]
    bounding_boxes = bounding_boxes.float()
    bounding_boxes[:, list(range(ndim)), :] = bounding_boxes[:, list(range(ndim))[::-1], :]
    scaler = bounding_boxes[:, :, 1] - bounding_boxes[:, :, 0]
    scaler = scaler.unsqueeze(1) * (torch.eye(ndim).type(scaler.type())).unsqueeze(0)
    shift = (-1.0 + bounding_boxes.sum(-1)).unsqueeze(-1)
    aff_matrices = torch.cat([scaler, shift], dim=-1)

    return aff_matrices


def roi_align(
    tensor: torch.Tensor,
    bounding_boxes: torch.Tensor,
    target_sizes: Union[List[int], Tuple[int]],
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: Optional[bool] = None,
) -> torch.Tensor:

    aff_matrix = compute_crop_resize_affine_matrix(bounding_boxes)
    grid = torch.nn.functional.affine_grid(aff_matrix, target_sizes)
    grid = grid.to(tensor.device)
    sample_tensor = torch.nn.functional.grid_sample(
        tensor, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners
    )
    return sample_tensor
