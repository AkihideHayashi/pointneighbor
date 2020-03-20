from ..type import Pnt, exp_pcl, VecSodAdj
from .. import functional as fn


# (n_bch, n_pnt, n_pnt, n_sft, n_dim)


def coo2_ful_simple(p: Pnt, rc: float):
    """Simple implementation for make coo-like 2-body problem.
    Make (n_bch, n_pnt, n_pnt, n_sft, n_dim) tensor and remove redundants.
    Low overhead but not efficient in terms of computational complexity.
    Use this function when your system is small and
    the size of lattice is almost uniform.
    This function is not efficient if your system is big or
    num_rpt is not uniform.
    """
    ep = exp_pcl(p)
    num_rpt = fn.minimum_neighbor(ep.cel_rec, ep.pbc, rc)
    max_rpt, _ = num_rpt.max(dim=0)
    sft_cel = fn.arange_prod(max_rpt * 2 + 1) - max_rpt
    sft_xyz = sft_cel.to(ep.cel_mat) @ ep.cel_mat
    _, n_pnt, _ = ep.pos_xyz.size()
    # (n_bch, n_pnt, n_pnt, n_sht, n_dim)
    pos_xyz_i = ep.pos_xyz[:, :, None, None, :]
    pos_xyz_j = ep.pos_xyz[:, None, :, None, :]
    sft_xyz_ij = sft_xyz[:, None, None, :, :]
    vec = fn.vector(pos_xyz_i, pos_xyz_j, sft_xyz_ij)
    sod = fn.square_of_distance(vec, dim=-1)

    ent_i = ep.ent[:, :, None, None]
    ent_j = ep.ent[:, None, :, None]

    val_ent = ent_i & ent_j
    val_idt = fn.val_idt(sft_cel, n_pnt)[None, :, :, :]
    val_pbc = fn.val_pbc(sft_cel, ep.pbc)[:, None, None, :]
    val_cut = sod <= rc * rc
    val = val_ent & val_idt & val_pbc & val_cut
    adj = fn.aranges_like(sod)
    ret = VecSodAdj(vec[val], sod[val], adj[:, val], sft_cel)
    return ret
