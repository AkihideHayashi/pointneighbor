from typing import NamedTuple
from torch import Tensor
from . import functional as fn


class Pnt(NamedTuple):
    cel: Tensor  # float [n_batch, n_dim, n_dim]  : cells
    pbc: Tensor  # bool  [n_batch, n_dim]         : periodic boundary contition
    pos: Tensor  # float [n_batch, n_point, n_dim]: positions (xyz)
    ent: Tensor  # bool  [n_batch, n_point]       : entity


class PntExp(NamedTuple):
    cel_mat: Tensor  # Pnt.cel
    cel_inv: Tensor  # Pnt.cel.inverse()
    cel_rec: Tensor  # Pnt.cel.inverse().t()
    pbc: Tensor      # Pnt.pbc
    pos_xyz: Tensor  # Pnt.pos
    pos_cel: Tensor  # Pnt.pos @ PntExp.cel_inv
    ent: Tensor      # Pnt.ent


class Adj(NamedTuple):
    adj: Tensor  # int [...]          : Adjacent
    sft: Tensor  # int [n_sft, n_dim] : shifts


class VecSodAdj(NamedTuple):
    vec: Tensor  # float [..., n_dim]
    sod: Tensor  # float [...]
    adj: Tensor  # float [...]
    sft: Tensor  # int   [n_sft, n_dim]


def contract(vsa: VecSodAdj, pe: PntExp, rc: float):
    n, i, j, _ = vsa.adj.unbind(0)
    ei = pe.ent[n, i]
    ej = pe.ent[n, j]
    val_cut = vsa.sod <= rc * rc
    val_ent = ei & ej
    val = val_cut & val_ent
    return VecSodAdj(vec=vsa.vec[val], sod=vsa.sod[val],
                     adj=vsa.adj[:, val], sft=vsa.sft)


def pnt_exp(p: Pnt):
    cel_mat, pbc, pos, ent = p
    cel_inv = cel_mat.inverse()
    cel_rec = cel_inv.transpose(-2, -1)
    pos_cel = fn.to_unit_cell(pos @ cel_inv, p.pbc)
    pos_xyz = pos_cel @ cel_mat
    return PntExp(
        cel_mat=cel_mat,
        cel_inv=cel_inv,
        cel_rec=cel_rec,
        pbc=pbc,
        pos_xyz=pos_xyz,
        pos_cel=pos_cel,
        ent=ent,
    )


def vec_sod_adj(pe: PntExp, adj: Adj, rc: float):
    cel = pe.cel_mat
    pos_xyz = pe.pos_xyz
    nijs, sft = adj
    n, i, j, s = nijs.unbind(0)
    sft_xyz = sft.to(cel) @ cel
    ri = pos_xyz[n, i, :]
    rj = pos_xyz[n, j, :]
    rs = sft_xyz[n, s, :]
    vec = fn.vector(ri, rj, rs)
    sod = fn.square_of_distance(vec, dim=-1)
    ent = pe.ent
    ei = ent[n, i]
    ej = ent[n, j]
    val_cut = sod <= rc * rc
    val_ent = ei & ej
    val = val_cut & val_ent
    return VecSodAdj(vec=vec[val], sod=sod[val], adj=nijs[:, val], sft=sft)


def vec_sod_adj_to_adj(vsa: VecSodAdj):
    return Adj(adj=vsa.adj, sft=vsa.sft)
