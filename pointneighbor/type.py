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


class AdjSftSpcVecSod(NamedTuple):
    adj: Tensor  # float [...]
    sft: Tensor  # int   [n_sft, n_dim]
    spc: Tensor  # int [n_bch, n_pnt: n_dim]
    vec: Tensor  # float [..., n_dim]
    sod: Tensor  # float [...]


def contract(vsa: AdjSftSpcVecSod, pe: PntExp, rc: float):
    n, i, j, _ = vsa.adj.unbind(0)
    ei = pe.ent[n, i]
    ej = pe.ent[n, j]
    val_cut = vsa.sod <= rc * rc
    val_ent = ei & ej
    val = val_cut & val_ent
    return AdjSftSpcVecSod(
        adj=vsa.adj[:, val], sft=vsa.sft, spc=vsa.spc,
        vec=vsa.vec[val], sod=vsa.sod[val],
    )


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


def vec_sod_adj(pe: PntExp, adj: AdjSftSpc, rc: float):
    cel = pe.cel_mat
    pos_xyz = pe.pos_xyz
    nijs = adj.adj
    sft = adj.sft
    n, i, j, s = nijs.unbind(0)
    sft_xyz = sft.to(cel) @ cel
    pos_xyz_unt = fn.to_unit_cell(pos_xyz, adj.spc @ pe.cel_mat)
    assert fn.in_unit_cell(pos_xyz_unt @ pe.cel_inv, pe.ent)
    ri = pos_xyz_unt[n, i, :]
    rj = pos_xyz_unt[n, j, :]
    rs = sft_xyz[n, s, :]
    vec = fn.vector(ri, rj, rs)
    sod = fn.square_of_distance(vec, dim=-1)
    ent = pe.ent
    ei = ent[n, i]
    ej = ent[n, j]
    val_cut = sod <= rc * rc
    val_ent = ei & ej
    val = val_cut & val_ent
    return AdjSftSpcVecSod(
        adj=nijs[:, val], sft=sft, spc=adj.spc,
        vec=vec[val], sod=sod[val],
    )


def vec_sod_adj_to_adj(vsa: AdjSftSpcVecSod):
    return AdjSftSpc(adj=vsa.adj, sft=vsa.sft, spc=vsa.spc)


def coo2_vec(adj: AdjSftSpc, pos: Tensor, cel: Tensor):
    n, i, j, s = adj.adj.unbind(0)
    pos_uni = fn.to_unit_cell(pos, adj.spc @ cel)
    sft = adj.sft.to(cel) @ cel
    ri = pos_uni[n, i, :]
    rj = pos_uni[n, j, :]
    rs = sft[n, s, :]
    return fn.vector(ri, rj, rs)
