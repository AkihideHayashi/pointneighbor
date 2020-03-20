from typing import NamedTuple
from torch import Tensor
from . import functional as fn


class Pnt(NamedTuple):
    cel: Tensor  # lattice float(n_batch, n_dim, n_dim)
    pbc: Tensor  # periodic boundary contition (n_batch, n_dim)
    pos: Tensor  # xyz-positions float(n_batch, n_point, n_dim)
    ent: Tensor  # entity bool(n_batch, n_point)


class PntExt(NamedTuple):
    cel_mat: Tensor
    cel_inv: Tensor
    cel_rec: Tensor
    pbc: Tensor
    pos_xyz: Tensor
    pos_cel: Tensor
    ent: Tensor


class Adj(NamedTuple):
    adj: Tensor
    sft: Tensor


class VecSodAdj(NamedTuple):
    vec: Tensor
    sod: Tensor
    adj: Tensor
    sft: Tensor


def exp_pcl(p: Pnt):
    cel_mat, pbc, pos_xyz, ent = p
    cel_inv = cel_mat.inverse()
    cel_rec = cel_inv.transpose(-2, -1)
    pos_cel = pos_xyz @ cel_inv
    return PntExt(
        cel_mat=cel_mat,
        cel_inv=cel_inv,
        cel_rec=cel_rec,
        pbc=pbc,
        pos_xyz=pos_xyz,
        pos_cel=pos_cel,
        ent=ent,
    )


def vec_sod_adj(pcl: Pnt, adj: Adj, rc: float):
    cel = pcl.cel
    pos_xyz = pcl.pos
    nijs, sft = adj
    n, i, j, s = nijs.unbind(0)
    sft_xyz = sft.to(cel) @ cel
    ri = pos_xyz[n, i, :]
    rj = pos_xyz[n, j, :]
    rs = sft_xyz[n, s, :]
    vec = fn.vector(ri, rj, rs)
    sod = fn.square_of_distance(vec, dim=-1)
    ent = pcl.ent
    ei = ent[n, i]
    ej = ent[n, j]
    val_cut = sod <= rc * rc
    val_ent = ei & ej
    val = val_cut & val_ent
    return VecSodAdj(vec=vec[val], sod=sod[val], adj=nijs[:, val], sft=sft)
