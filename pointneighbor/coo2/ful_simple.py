from ..type import PntFul, AdjSftSpc
from .. import functional as fn


# (n_bch, n_pnt, n_pnt, n_sft, n_dim)
def coo2_ful_simple(pe: PntFul, rc: float) -> AdjSftSpc:
    """Simple implementation for make coo-like 2-body problem.
    Make (n_bch, n_pnt, n_pnt, n_sft, n_dim) tensor and remove redundants.
    Low overhead but not efficient in terms of computational complexity.
    Use this function when your system is small and
    the size of lattice is almost uniform.
    This function is not efficient if your system is big or
    num_rpt is not uniform.
    """
    num_rpt = fn.minimum_neighbor(pe.cel_rec.detach(), pe.pbc, rc)
    max_rpt, _ = num_rpt.max(dim=0)
    sft_cel = fn.arange_prod(max_rpt * 2 + 1) - max_rpt
    sft_xyz = sft_cel.to(pe.cel_mat) @ pe.cel_mat.detach()
    _, n_pnt, _ = pe.pos_xyz.size()
    # (n_bch, n_pnt, n_pnt, n_sht, n_dim)
    pos_xyz = fn.to_unit_cell(pe.pos_xyz.detach(), pe.sft_xyz.detach())
    pos_xyz_i = pos_xyz[:, :, None, None, :]
    pos_xyz_j = pos_xyz[:, None, :, None, :]
    sft_xyz_ij = sft_xyz[:, None, None, :, :]
    assert not pos_xyz.requires_grad
    assert not pos_xyz_i.requires_grad
    assert not pos_xyz_j.requires_grad
    assert not sft_xyz_ij.requires_grad
    vec = fn.vector(pos_xyz_i, pos_xyz_j, sft_xyz_ij)
    sod = fn.square_of_distance(vec, dim=-1)
    assert not vec.requires_grad
    assert not sod.requires_grad

    ent_i = pe.ent[:, :, None, None]
    ent_j = pe.ent[:, None, :, None]

    val_ent = ent_i & ent_j
    val_idt = fn.val_idt(sft_cel, n_pnt)[None, :, :, :]
    val_pbc = fn.val_pbc(sft_cel, pe.pbc)[:, None, None, :]
    val_cut = sod <= rc * rc
    val = val_ent & val_idt & val_pbc & val_cut
    adj = fn.aranges_like(sod)
    ret = AdjSftSpc(
        adj=adj[:, val], sft=sft_cel, spc=pe.sft_cel,
    )
    return ret
