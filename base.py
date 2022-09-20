import json
import numbers
import random

from typing import Any
from typing import Dict
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union


import numpy as np
import torch


class BaseTransform:
    """
    This class serves as the base class for all transforms.
    It supports applying transforms with predefined transform parameters
    or using randomly generated transform parameters.
    """

    def __init__(
        self,
        p: float = 0.5,
        always_apply: bool = False,
        accept_types: Optional[Set[Any]] = None,
        exclude_types: Optional[Set[Any]] = None,
        freeze_param: bool = False,
    ):
        """
        Args:
            p: probability threshold deciding whether this transform will be applied or not. Default 0.5.
            always_apply: True if the transform will always be applied, False if not going to be applied.
            accept_types: the transform will be applied only for data types in the accept_types.
            exclude_types: the transform will not be applied for data types in the exclude_types.
            freeze_param: True if the transform will be applied with cached parameters (without regenerated randomly).

        """
        self.accept_types = {torch.Tensor, np.ndarray} if accept_types is None else accept_types
        self.exclude_types = {str} if exclude_types is None else exclude_types
        self.p = p
        self.always_apply = always_apply
        self.freeze_param = freeze_param
        self.params: Dict[str, Any] = {}

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=4)

    def to_dict(self) -> Dict[str, Any]:
        state = {"__class_fullname__": self.get_class_fullname()}
        state.update(self.get_base_init_args())
        state.update(self.get_transform_init_args())
        state.update({"randomized_params": self.params})
        return state

    def get_transform_init_args(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.get_transform_init_args_names()}

    def get_base_init_args(self) -> Dict[str, Any]:
        return {"always_apply": self.always_apply, "p": self.p}

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        raise NotImplementedError(
            f"Class {self.get_class_fullname()} is not serializable because the "
            f"`get_transform_init_args_names` method is not "
            "implemented"
        )

    def get_params(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        This function should be implemented by the subclass to provide a way
        to generate transform parameters.
        Args:
            data_dict: the input data dictionary on which the transform will be applied, some transforms
            require knowing the data shape in order to generate function parameters.
        """
        return {}

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.freeze_param:
            return self.apply_with_params(self.params, data_dict)

        if random.random() < self.p or self.always_apply:
            params = self.get_params(data_dict)
            return self.apply_with_params(params, data_dict)

        return data_dict

    def apply_with_params(self, params: Dict[str, Any], data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            params: parameters used in the transform function.
            data_dict: the input data dictionary on which the transform will be applied,
            we conduct data type checks using the ``"accept_types"`` and ``"exclude_types"``.
        """
        self.params.update(params)
        res = {}
        for key, data in data_dict.items():
            if isinstance(data, tuple(self.accept_types)) and not isinstance(data, tuple(self.exclude_types)):
                res[key] = self.apply_function_on_key(key, data)
            else:
                res[key] = data_dict[key]
        return res

    def apply_to_image(self, data: Any):
        raise NotImplementedError(f"Method apply_to_image is not implemented in class {self.__class__.__name__}")

    def apply_to_mask(self, data: Any):
        raise NotImplementedError(f"Method apply_to_mask is not implemented in class {self.__class__.__name__}")

    def apply_to_box(self, data: Any):
        raise NotImplementedError(f"Method apply_to_box is not implemented in class {self.__class__.__name__}")

    def apply_to_point_cloud(self, data_dict: Any):
        raise NotImplementedError(f"Method apply_to_point_cloud is not implemented in class {self.__class__.__name__}")

    def apply_function_on_key(self, key: str, data: Any):
        """
        We apply functions on data according to its key (representing the semantic data type).
        """
        if "image" in key:
            data = self.apply_to_image(data)
        elif "mask" in key:
            data = self.apply_to_mask(data)
        elif "box" in key:
            data = self.apply_to_box(data)
        elif "points" in key:
            data = self.apply_to_point_cloud(data)
        else:
            pass
        return data

    def check_positive_numbers_or_range(
        self,
        value: Union[numbers.Number, Tuple[numbers.Number, numbers.Number]],
        name: str,
        bounds: Tuple[numbers.Number, numbers.Number] = (0.0, float("inf")),
    ) -> Union[numbers.Number, Tuple[numbers.Number, numbers.Number]]:
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
                raise ValueError(
                    f"{name} values should be between {bounds}, "
                    "and lowerbound should be smaller or equal to the upperbound."
                )
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

        return value

    def check_numbers_or_tuple(
        self,
        value: Union[numbers.Number, Tuple[numbers.Number, numbers.Number]],
        name: str,
        bounds: Tuple[numbers.Number, numbers.Number] = (0.0, float("inf")),
    ) -> Union[numbers.Number, Tuple[numbers.Number, numbers.Number]]:
        if isinstance(value, numbers.Number):
            if not bounds[0] <= value <= bounds[1]:
                raise ValueError(f"If {name} is a single number, it must be between bounds {bounds}.")
        elif isinstance(value, (tuple, list)):
            if not all(bounds[0] <= v <= bounds[1] for v in value):
                raise ValueError(f"{name} values should be between {bounds}")
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple of numbers.")

        return value

    def check_dimension_index(self, value: int, name: str, accept_indices: Tuple[int] = (0, 1, 2, -1)) -> int:
        if isinstance(value, (int,)):
            if value not in accept_indices:
                raise ValueError(f"{name} only support dimension index among {accept_indices}.")
        else:
            raise TypeError(f"{name} should be a single integer among {accept_indices}.")

        return value

    def check_range(
        self, value: Tuple[numbers.Number, numbers.Number], name: str
    ) -> Tuple[numbers.Number, numbers.Number]:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            if value[0] > value[1]:
                raise ValueError("value[0] should be no larger than value[1]")
        else:
            raise TypeError(f"{name} should be a tuple or a list of two numbers among.")

        return value

    def check_positive_range(
        self, value: Tuple[numbers.Number, numbers.Number], name: str
    ) -> Tuple[numbers.Number, numbers.Number]:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            if not 0 <= value[0] <= value[1]:
                raise ValueError("value[0] should be not larger than value[1] and both are positive numbers.")
        else:
            raise TypeError(f"{name} should be a tuple or a list of two positive numbers.")

        return value

    @classmethod
    def get_class_fullname(cls) -> str:
        return f"{cls.__module__}.{cls.__name__}"


class NumpyToTensor(BaseTransform):
    """
    This transform convert numpy.ndarray in the data to torch.Tensor.
    """

    def __init__(self):
        super(NumpyToTensor, self).__init__(1.0, True)
        self._accept_types = {np.ndarray}

    def apply_function_on_key(self, key: str, data: np.ndarray):
        return torch.as_tensor(data)


class TensorToNumpy(BaseTransform):
    """
    This transform convert torch.Tensor in the data to numpy.ndarray.
    """

    def __init__(self):
        super(TensorToNumpy, self).__init__(1.0, True)
        self._accept_types = {torch.Tensor}

    def apply_function_on_key(self, key: str, data: torch.Tensor):
        return data.numpy()
