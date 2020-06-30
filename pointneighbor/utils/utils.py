from typing import Tuple
import torch
from torch import Tensor
from ..types import AdjSftSpc, VecSod, is_coo2, is_lil2
from .. import functional as fn


def cutoff_coo2(adj: AdjSftSpc, vec_sod: VecSod, rc: float
                ) -> Tuple[AdjSftSpc, VecSod]:
    """Cutoff coo2 adjacent information."""
    sod = vec_sod.sod
    vec = vec_sod.vec
    val = sod <= rc * rc
    ret_adj = AdjSftSpc(adj=adj.adj[:, val], sft=adj.sft, spc=adj.spc)
    ret_sod = VecSod(vec=vec[val, :], sod=sod[val])
    return ret_adj, ret_sod


def coo2_vec(adj: AdjSftSpc, pos: Tensor, cel: Tensor):
    assert is_coo2(adj)
    n, i, j, s = adj.adj.unbind(0)
    pos_uni = fn.to_unit_cell(pos, adj.spc @ cel)
    sft = adj.sft.to(cel) @ cel
    ri = pos_uni[n, i, :]
    rj = pos_uni[n, j, :]
    rs = sft[n, s, :]
    return fn.vector(ri, rj, rs)


def coo2_vec_sod(adj: AdjSftSpc, pos: Tensor, cel: Tensor):
    vec = coo2_vec(adj, pos, cel)
    sod = fn.square_of_distance(vec)
    return VecSod(vec=vec, sod=sod)


def lil2_vec(adj: AdjSftSpc, pos: Tensor, cel: Tensor):
    assert is_lil2(adj)
    pos_uni = fn.to_unit_cell(pos, adj.spc @ cel)
    sft = adj.sft.to(cel) @ cel
    j, s = adj.adj.unbind(0)
    n_bch = pos.size(0)
    i = torch.arange(n_bch, device=pos.device, dtype=torch.long)[:, None, None]
    rj = pos_uni[i, j[:, :, :], :]
    ri = pos_uni[:, :, None, :]
    rs = sft[i, s[:, :, :], :]
    vec = fn.vector(ri, rj, rs)
    return torch.where(
        j[:, :, :, None].expand_as(vec) >= 0, vec, torch.zeros_like(vec))


def lil2_vec_sod(adj: AdjSftSpc, pos: Tensor, cel: Tensor):
    vec = lil2_vec(adj, pos, cel)
    sod = fn.square_of_distance(vec)
    return VecSod(vec=vec, sod=sod)
