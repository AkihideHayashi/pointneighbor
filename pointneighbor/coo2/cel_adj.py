from typing import NamedTuple, Optional
from torch import Tensor
import torch
from .. import functional as fn
from ..type import Pnt, PntExt, exp_pcl


class CelAdj(NamedTuple):
    adj: Tensor
    sft: Tensor
    div: Tensor


def number_division(cel_mat: Tensor, rc: float):
    ndiv = ((cel_mat / rc).norm(p=2, dim=-1) - 1e-4).floor().to(torch.int64)
    return ndiv


def cel_adj(p: Pnt, rc: float, num_div: Optional[Tensor] = None):
    ep: PntExt = exp_pcl(p)
    if num_div is None:
        num_div = number_division(ep.cel_mat, rc)
    max_div, _ = num_div.max(0)
    pbc = ep.pbc
    num_adj = fn.minimum_neighbor(ep.cel_rec * num_div[:, None, :], pbc, rc)
    max_adj, _ = num_adj.max(0)
    pcl = fn.arange_prod(max_div)  # parcel
    sft_pcl = fn.arange_prod(max_adj * 2 + 1) - max_adj

    n_sft = sft_pcl.size()[0]
    n_bch, _, n_dim = ep.pos_xyz.size()
    n_pcl = pcl.size()[0]

    size = [n_bch, n_pcl, n_sft, n_dim]
    pcl_i = pcl[None, :, None, :].expand(size)
    pcl_j = (pcl[None, :, None, :] + sft_pcl[None, None, :, :]).expand(size)
    pcl_mod_j = pcl_j % num_div[:, None, None, :]
    sft_pcl_cel = torch.where(
        pcl_j >= 0,
        pcl_j / num_div[:, None, None, :],
        (pcl_j + 1) / num_div[:, None, None, :] - 1,
    )

    val_pcl = _val_pcl(pcl_i, num_div) & _val_pcl(pcl_mod_j, num_div)
    val_sft_pcl_cel = ((sft_pcl_cel == 0) | pbc[:, None, None, :]).all(-1)
    val = val_pcl & val_sft_pcl_cel
    n = fn.arange_like(val, 0)[val]
    i = fn.ravel(pcl_i, num_div[:, None, None, :], -1)[val]
    j = fn.ravel(pcl_mod_j, num_div[:, None, None, :], -1)[val]
    num_rpt, _ = sft_pcl_cel.abs().flatten(0, 2).max(dim=0)
    s = fn.ravel1(sft_pcl_cel + num_rpt, num_rpt * 2 + 1, -1)[val]
    sft = fn.arange_prod(num_rpt * 2 + 1) - num_rpt
    return CelAdj(adj=torch.stack([n, i, j, s]), sft=sft, div=num_div)


def _val_pcl(pcl, num_div):
    return (pcl < num_div[:, None, None, :]).all(dim=-1)
