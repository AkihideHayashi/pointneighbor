import torch
from torch import Tensor
from .. import functional as fn


def coo3(neighbor: Tensor) -> Tensor:
    # num is number of bond for each atoms.
    # For example, if num == [2, 3]
    # make (2, 3, 3) tensor and remove |i, j| and (i, i).

    # (0 0) (0 1) |0 2|
    # (1 0) (1 1) |1 2|
    # |2 0| |2 1| |2 2|

    # (0 0) (0 1) (0 2)
    # (1 0) (1 1) (1 2)
    # (2 0) (2 1) (2 2)
    n2 = neighbor.detach()[0]
    i2 = neighbor.detach()[1]
    i2_max = int(i2.max().item()) if len(i2) > 0 else 0
    ni2, _ = (i2 + (i2_max + 1) * n2).sort()

    _, _, num = torch.unique_consecutive(
        ni2, return_inverse=True, return_counts=True)
    n = len(num)
    m = int(num.max().item()) if len(num) > 0 else 0
    idx = torch.arange(m, dtype=n2.dtype, device=n2.device)

    f = idx[None, :] < num[:, None]
    filt = (f[:, :, None] & f[:, None, :] &
            (idx[None, :, None] != idx[None, None, :]))

    base = fn.cumsum_from_zero(num)[torch.repeat_interleave(num * num - num)]
    j3 = idx[None, :, None].expand([n, m, m])[filt] + base
    k3 = idx[None, None, :].expand([n, m, m])[filt] + base
    return torch.stack([j3, k3])
