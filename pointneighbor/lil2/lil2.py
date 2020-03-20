from typing import List
import torch
from torch import Tensor
from ..type import Adj
from .. import functional as fn


def lil2(adj_coo: Adj, size: List[int]):
    nijs, sft = adj_coo
    _, _, j, s = nijs.unbind(0)
    val = mask_coo_to_lil(adj_coo, size)
    ret_s = coo_to_lil(s, val, 0)
    ret_j = coo_to_lil(j, val, -1)
    ret = torch.stack([ret_j, ret_s])
    return Adj(adj=ret, sft=sft)


def mask_coo_to_lil(adj_coo: Adj, size: List[int]):
    n_bch, n_pnt = size
    nijs = adj_coo.adj
    n, i, _, _ = nijs.unbind(0)
    ni = n * n_pnt + i
    cnt = fn.count_number(ni, [n_bch * n_pnt]).view([n_bch, n_pnt])
    max_cnt = int(cnt.max().item())
    mask = fn.arange([n_bch, n_pnt, max_cnt], -1) < cnt[:, :, None]
    return mask


def coo_to_lil(coo: Tensor, mask: Tensor, dummy: int):
    size = list(mask.size())
    for s in coo.size()[1:]:
        size.append(s)
    ret = torch.ones(size, device=coo.device, dtype=coo.dtype) * dummy
    ret[mask] = coo
    return ret
