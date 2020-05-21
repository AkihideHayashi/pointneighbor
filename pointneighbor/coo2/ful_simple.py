from ..type import PntExp, AdjSftSizVecSod
from .. import functional as fn


def coo2_ful_simple(pe: PntExp, rc: float) -> AdjSftSizVecSod:
    """Simple implementation for make coo-like 2-body problem.
    Make (n_bch, n_pnt, n_pnt, n_sft, n_dim) tensor and remove redundants.
    Low overhead but not efficient in terms of computational complexity.
    Use this function when your system is small and
    the size of lattice is almost uniform.
    This function is not efficient if your system is big or
    num_rpt is not uniform.
    """
    requires_grad = False
    for prop in pe:
        requires_grad = prop.requires_grad or requires_grad
    if not requires_grad:
        return coo2_ful_simple_nograd(pe, rc)
    else:
        return coo2_ful_simple_grad(pe, rc)


# (n_bch, n_pnt, n_pnt, n_sft, n_dim)
def coo2_ful_simple_nograd(pe: PntExp, rc: float) -> AdjSftSizVecSod:
    """Simple implementation for make coo-like 2-body problem.
    Make (n_bch, n_pnt, n_pnt, n_sft, n_dim) tensor and remove redundants.
    Low overhead but not efficient in terms of computational complexity.
    Use this function when your system is small and
    the size of lattice is almost uniform.
    This function is not efficient if your system is big or
    num_rpt is not uniform.
    """
    num_rpt = fn.minimum_neighbor(pe.cel_rec, pe.pbc, rc)
    max_rpt, _ = num_rpt.max(dim=0)
    sft_cel = fn.arange_prod(max_rpt * 2 + 1) - max_rpt
    sft_xyz = sft_cel.to(pe.cel_mat) @ pe.cel_mat
    _, n_pnt, _ = pe.pos_xyz.size()
    # (n_bch, n_pnt, n_pnt, n_sht, n_dim)
    pos_xyz_i = pe.pos_xyz[:, :, None, None, :]
    pos_xyz_j = pe.pos_xyz[:, None, :, None, :]
    sft_xyz_ij = sft_xyz[:, None, None, :, :]
    vec = fn.vector(pos_xyz_i, pos_xyz_j, sft_xyz_ij)
    sod = fn.square_of_distance(vec, dim=-1)

    ent_i = pe.ent[:, :, None, None]
    ent_j = pe.ent[:, None, :, None]

    val_ent = ent_i & ent_j
    val_idt = fn.val_idt(sft_cel, n_pnt)[None, :, :, :]
    val_pbc = fn.val_pbc(sft_cel, pe.pbc)[:, None, None, :]
    val_cut = sod <= rc * rc
    val = val_ent & val_idt & val_pbc & val_cut
    adj = fn.aranges_like(sod)
    ret = AdjSftSizVecSod(
        adj=adj[:, val], sft=sft_cel, siz=list(pe.ent.size()),
        vec=vec[val], sod=sod[val],
    )
    return ret


def coo2_ful_simple_grad(pe: PntExp, rc: float) -> AdjSftSizVecSod:
    """Simple implementation for make coo-like 2-body problem.
    Make (n_bch, n_pnt, n_pnt, n_sft, n_dim) tensor and remove redundants.
    Low overhead but not efficient in terms of computational complexity.
    Use this function when your system is small and
    the size of lattice is almost uniform.
    This function is not efficient if your system is big or
    num_rpt is not uniform.
    """
    num_rpt = fn.minimum_neighbor(pe.cel_rec.detach(), pe.pbc, rc)
    assert not num_rpt.requires_grad
    max_rpt, _ = num_rpt.max(dim=0)
    assert not max_rpt.requires_grad
    sft_cel = fn.arange_prod(max_rpt * 2 + 1) - max_rpt
    assert not sft_cel.requires_grad
    sft_xyz = sft_cel.to(pe.cel_mat) @ pe.cel_mat
    _, n_pnt, _ = pe.pos_xyz.size()
    # (n_bch, n_pnt, n_pnt, n_sht, n_dim)
    pos_xyz_i = pe.pos_xyz.detach()[:, :, None, None, :]
    pos_xyz_j = pe.pos_xyz.detach()[:, None, :, None, :]
    sft_xyz_ij = sft_xyz.detach()[:, None, None, :, :]
    vec = fn.vector(pos_xyz_i, pos_xyz_j, sft_xyz_ij)
    sod = fn.square_of_distance(vec, dim=-1)
    assert not vec.requires_grad

    ent_i = pe.ent[:, :, None, None]
    ent_j = pe.ent[:, None, :, None]

    val_ent = ent_i & ent_j
    val_idt = fn.val_idt(sft_cel, n_pnt)[None, :, :, :]
    val_pbc = fn.val_pbc(sft_cel, pe.pbc)[:, None, None, :]
    val_cut = sod <= rc * rc
    val = val_ent & val_idt & val_pbc & val_cut
    adj = fn.aranges_like(sod)

    nijs = adj[:, val]
    n, i, j, s = nijs.unbind(0)
    ri = pe.pos_xyz[n, i]
    rj = pe.pos_xyz[n, j]
    rs = sft_xyz[n, s]
    vec = fn.vector(ri, rj, rs)
    sod = fn.square_of_distance(vec, dim=-1)
    ret = AdjSftSizVecSod(
        adj=nijs, sft=sft_cel, siz=list(pe.ent.size()),
        vec=vec, sod=sod,
    )
    return ret
