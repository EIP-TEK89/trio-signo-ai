import random
import torch


def rand_fix_interval(limit: float) -> float:
    return random.uniform(-limit, limit)


def from_1d_tensor_to_list_int(tensor: torch.Tensor) -> list[int]:
    """
    Convert a 1D tensor to a list of integers.
    """
    if tensor.dim() != 1:
        raise ValueError("Tensor must be 1D")
    return tensor.tolist()


def from_1d_tensor_to_list_float(tensor: torch.Tensor) -> list[float]:
    """
    Convert a 1D tensor to a list of floats.
    """
    if tensor.dim() != 1:
        raise ValueError("Tensor must be 1D")
    return tensor.tolist()
