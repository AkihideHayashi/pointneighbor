from typing import NamedTuple
from torch import Tensor
from . import functional as fn


class Pnt(NamedTuple):
    """The standard, easy-to-use input format.
    cel (float[n_bch, n_dim, n_dim]) : The cell vectors.
    pbc (bool[n_bch, n_dim]) : periodic boundary condition.
    pos (float[n_bch, n_pnt, n_dim]) : positions (xyz form)
    ent (bool[n_bch, n_pnt]) : entity. (not dummy.)
    """
    cel: Tensor
    pbc: Tensor
    pos: Tensor
    ent: Tensor


class PntExp(NamedTuple):
    """Expanded Point information.
    The standerd full information struct to be used inside pointneighbor.
    """
    cel_mat: Tensor  # Pnt.cel
    cel_inv: Tensor  # Pnt.cel.inverse()
    cel_rec: Tensor  # Pnt.cel.inverse().t()
    pbc: Tensor      # Pnt.pbc
    pos_xyz: Tensor  # Pnt.pos
    pos_cel: Tensor  # Pnt.pos @ PntExp.cel_inv
    ent: Tensor      # Pnt.ent
    sft_xyz: Tensor  # sft_xyz
    sft_cel: Tensor  # sft_cel


class AdjSftSpc(NamedTuple):
    adj: Tensor  # int [...]          : Adjacent
    sft: Tensor  # int [n_sft, n_dim] : shifts
    spc: Tensor  # int [n_bch, n_pnt: n_dim]


class VecSod(NamedTuple):
    """Vector and Square of distance."""
    vec: Tensor
    sod: Tensor


def pnt(cel: Tensor, pbc: Tensor, pos: Tensor, ent: Tensor):
    return Pnt(cel=cel, pbc=pbc, pos=pos, ent=ent)


def pnt_exp(p: Pnt):
    cel_mat, pbc, pos_xyz, ent = p
    cel_inv = cel_mat.inverse()
    cel_rec = cel_inv.transpose(-2, -1)
    pos_cel = pos_xyz @ cel_inv
    sft_cel = fn.get_pos_sft(pos_cel, p.pbc)
    sft_xyz = sft_cel @ cel_mat
    return PntExp(
        cel_mat=cel_mat,
        cel_inv=cel_inv,
        cel_rec=cel_rec,
        pbc=pbc,
        pos_xyz=pos_xyz,
        pos_cel=pos_cel,
        ent=ent,
        sft_xyz=sft_xyz,
        sft_cel=sft_cel,
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
