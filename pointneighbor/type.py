from typing import NamedTuple
import warnings
import torch
from torch import Tensor
from . import functional as fn


class PntFul(NamedTuple):
    """Full spec Point information.
    The standerd full information struct to be used inside pointneighbor.
    """
    cel_mat: Tensor  # Pnt.cel
    cel_inv: Tensor  # Pnt.cel.inverse()
    cel_rec: Tensor  # Pnt.cel.inverse().t()
    pbc: Tensor      # Pnt.pbc
    pos_xyz: Tensor  # Pnt.pos
    pos_cel: Tensor  # Pnt.pos @ PntExp.cel_inv
    ent: Tensor      # Pnt.ent
    spc_xyz: Tensor  # spc_xyz  # もしかして、divとかpclの方がいいかも
    spc_cel: Tensor  # spc_cel


class AdjSftSpc(NamedTuple):
    adj: Tensor  # int [...]          : Adjacent
    sft: Tensor  # int [n_sft, n_dim] : shifts
    spc: Tensor  # int [n_bch, n_pnt: n_dim]


class VecSod(NamedTuple):
    """Vector and Square of distance."""
    vec: Tensor
    sod: Tensor


def pnt_ful(cel: Tensor, pbc: Tensor, pos: Tensor, ent: Tensor):
    cel_mat = cel
    pos_xyz = pos
    cel_inv = cel_mat.inverse()
    cel_rec = cel_inv.transpose(-2, -1)
    pos_cel = pos_xyz @ cel_inv
    spc_cel = fn.get_pos_spc(pos_cel, pbc)
    spc_xyz = spc_cel @ cel_mat
    return PntFul(
        cel_mat=cel_mat,
        cel_inv=cel_inv,
        cel_rec=cel_rec,
        pbc=pbc,
        pos_xyz=pos_xyz,
        pos_cel=pos_cel,
        ent=ent,
        spc_xyz=spc_xyz,
        spc_cel=spc_cel,
    )


def coo2_vec(adj: AdjSftSpc, pos: Tensor, cel: Tensor):
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


def coo2_adj_vec_sod(adj: AdjSftSpc, pos: Tensor, cel: Tensor, rc: float):
    vec, sod = coo2_vec_sod(adj, pos, cel)
    val = sod <= rc * rc
    ret_adj = AdjSftSpc(adj=adj.adj[:, val], sft=adj.sft, spc=adj.spc)
    ret_sod = VecSod(vec=vec[val, :], sod=sod[val])
    return ret_adj, ret_sod


def get_n_i_j(adj: AdjSftSpc):
    if adj.adj.size(0) != 4:
        raise RuntimeError()
    n, i, j, _ = adj.adj.unbind(0)
    return n, i, j


def get_n_i_j_s(adj: AdjSftSpc):
    warnings.warn('get_n_i_j_s is not tested.')
    if adj.adj.size(0) != 4:
        raise RuntimeError()
    n, i, j, s = adj.adj.unbind(0)
    spc = fn.to_unit_cell(torch.zeros_like(adj.spc), adj.spc)
    spc_i = spc[n, i]
    spc_j = spc[n, j]
    sft = fn.vector(spc_i, spc_j, adj.sft[n, s])
    return n, i, j, sft


def get_lil2_j_s(adj: AdjSftSpc):
    warnings.warn('get_lil2 is not tested.')
    if adj.adj.size(0) != 2:
        raise RuntimeError()
    j, s = adj.adj.unbind(0)
    assert j.dim() == 3  # n_bch, n_pnt, n_cnt
    i = fn.arange_like(j, 1)
    n = fn.arange_like(j, 0)
    spc = fn.to_unit_cell(torch.zeros_like(adj.spc), adj.spc)
    spc_i = spc[n, i]
    spc_j = spc[n, j]
    sft = fn.vector(spc_i, spc_j, adj.sft[s])
    return j, sft
