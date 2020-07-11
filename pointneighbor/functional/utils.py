"""Functions that compensate for the functions implemented in torch"""
from typing import List
import torch
from torch.jit import script
from torch import Tensor


def unsqueeze_like_(tensor: Tensor, dims: List[int], dist: Tensor):
    """Unsqueeze the tensor to have the same dimensions as dist.
    The dims dimension of result is the dimension of the original tensor. """
    assert tensor.dim() == len(dims)
    unsqueeze_(tensor, dims, dist.dim())


def unsqueeze_like(tensor: Tensor, dims: List[int], dist: Tensor):
    """out-place version of unsqueeze_like"""
    result = tensor.clone()
    unsqueeze_(result, dims, dist.dim())
    return result


def unsqueeze_(tensor: Tensor, dims: List[int], dim: int):
    """Unsqueeze the tensor to have the same dimensions as dim.
    The dims dimension of result is the dimension of the original tensor. """
    dims_: List[int] = []
    for d in dims:
        dims_.append(d % dim)
    for i in range(dim):
        if i not in dims_:
            tensor.unsqueeze_(i)


def unsqueeze(tensor: Tensor, dims: List[int], dim: int):
    """out-place version of unsqueeze_"""
    result = tensor.clone()
    unsqueeze_(result, dims, dim)
    return result


def expand_as_besides(tensor: Tensor, dims: List[int], dist: Tensor):
    """Expand dimensions other than the specified become same as dist."""
    shape = list(dist.size())
    for dim in dims:
        shape[dim] = tensor.size()[dim]
    return tensor.expand(shape)


def arange(size: List[int], dim: int, device_tensor: Tensor) -> Tensor:
    dim = dim % len(size)
    arange_ = torch.arange(size[dim],
                           dtype=torch.int64, device=device_tensor.device)
    unsqueeze_(arange_, [dim], len(size))
    return arange_.expand(size)


def arange_like(tensor: Tensor, dim: int) -> Tensor:
    """like torch.arrange for the specified dim with the same size as dist"""
    dim = dim % tensor.dim()
    arange_ = torch.arange(tensor.shape[dim]).to(
        device=tensor.device, dtype=torch.int64)
    unsqueeze_like_(arange_, [dim], tensor)
    return arange_.expand_as(tensor)


def aranges(size: List[int], device_tensor: Tensor) -> Tensor:
    return torch.stack([arange(size, i, device_tensor)
                        for i, _ in enumerate(size)])


def aranges_like(tensor: Tensor):
    size = list(tensor.size())
    return aranges(size, tensor).to(device=tensor.device, dtype=torch.int64)


def cumsum_from_zero(inp: Tensor, dim: int = 0):
    """Like torch.cumsum, but start from 0."""
    cumsum = torch.cumsum(inp, dim=dim)
    narrow = cumsum.narrow(dim=dim, start=0, length=max(0, inp.shape[dim] - 1))
    zero = torch.zeros([1]).to(inp)
    unsqueeze_like_(zero, [dim], inp)
    zero = expand_as_besides(zero, dims=[dim], dist=inp)
    return torch.cat([zero, narrow], dim=dim)


def arange_prod(dims: Tensor):
    """Arange cartesian prod for each dimension."""
    assert dims.dim() == 1, dims
    aranges_ = [torch.arange(d.item(), device=d.device) for d in dims]
    return cartesian_prod(aranges_)


def ravel1(index: Tensor, size: Tensor, dim: int):
    """Transform multi-index to single-index.

    Parameters:
        index: multi-index
        size: Sizes of each dimension.
        dim: Dimension for multi.
    """
    assert size.dim() == 1
    mag = size.flip(0).cumprod(0).flip(0).roll(-1, dims=0)
    mag[-1] = 1
    mag = unsqueeze(mag, [dim], index.dim())
    return (index * mag).sum(dim=dim)


def ravel(index: Tensor, size: Tensor, dim: int):
    """Transform multi-index to single-index.

    Parameters:
        index: multi-index
        size: Sizes of each dimension.
        dim: Dimension for multi.
    """
    assert index.dim() == size.dim()
    mag = size.flip(dim).cumprod(dim).flip(dim).roll(-1, dims=dim)
    idx = torch.tensor([mag.size()[dim]]).to(mag) - 1
    mag.index_fill_(-1, idx, 1)
    return (index * mag).sum(dim=dim)


@script
def cartesian_prod(inputs: List[Tensor]) -> Tensor:
    if len(inputs) == 1:
        return torch.cartesian_prod(inputs)[:, None]
    else:
        return torch.cartesian_prod(inputs)


def where_int(condition: Tensor, if_true: Tensor, if_false: int) -> Tensor:
    return torch.where(condition, if_true, torch.ones_like(if_true) * if_false)


def count_number(sorted_tensor, size: List[int]):
    unique, _, count = torch.unique_consecutive(sorted_tensor, True, True)
    dest = torch.zeros(size).to(unique)
    return torch.scatter(dest, 0, unique, count)


def replace_dummy(tensor: Tensor, dummy: int, to: int):
    return torch.where(tensor == dummy, torch.ones_like(tensor) * to, tensor)
