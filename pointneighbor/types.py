from typing import NamedTuple, Optional
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


def is_coo2(adj: AdjSftSpc):
    # n, i, j, s
    return (adj.adj.size(0) == 4) and (adj.adj.dim() == 2)


def is_lil2(adj: AdjSftSpc):
    # adj[n, i, 0] == j
    # adj[n, i, 1] == s
    return (adj.adj.size(0) == 2) and (adj.adj.dim() == 4)


class VecSod(NamedTuple):
    """Vector and Square of distance."""
    vec: Tensor
    sod: Tensor


def pnt_ful(cel_mat: Tensor, pbc: Tensor, pos_xyz: Tensor, ent: Tensor,
            cel_inv: Optional[Tensor] = None, cel_rec: Optional[Tensor] = None,
            pos_cel: Optional[Tensor] = None):
    if cel_inv is None:
        cel_inv = cel_mat.inverse()
    if cel_rec is None:
        cel_rec = cel_inv.transpose(-2, -1)
    if pos_cel is None:
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
