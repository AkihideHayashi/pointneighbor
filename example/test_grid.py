from math import pi
import torch
from torch import Tensor
from pointneighbor import functional as fn, PntFul, AdjSftSpc
import pointneighbor as pn
from common import CellParameter, triu_cell


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
    print(f'{num_rpt=}')
    max_rpt, _ = num_rpt.max(dim=0)
    sft_cel = fn.arange_prod(max_rpt * 2 + 1) - max_rpt
    sft_xyz = sft_cel.to(pe.cel_mat) @ pe.cel_mat.detach()
    _, n_pnt, _ = pe.pos_xyz.size()
    # (n_bch, n_pnt, n_pnt, n_sht, n_dim)
    pos_xyz = fn.to_unit_cell(pe.pos_xyz.detach(), pe.spc_xyz.detach())
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
        adj=adj[:, val], sft=sft_cel, spc=pe.spc_cel,
    )
    return ret


# (n_bch, n_pnt, n_pnt, n_sft, n_dim)
def coo2_ful_simple_2(pe: PntFul, rc: float) -> AdjSftSpc:
    """Simple implementation for make coo-like 2-body problem.
    Make (n_bch, n_pnt, n_pnt, n_sft, n_dim) tensor and remove redundants.
    Low overhead but not efficient in terms of computational complexity.
    Use this function when your system is small and
    the size of lattice is almost uniform.
    This function is not efficient if your system is big or
    num_rpt is not uniform.
    """
    # num_rpt = fn.minimum_neighbor(pe.cel_rec.detach(), pe.pbc, rc)
    num_rpt = get_minimum_radial_grid(pe.cel_rec, pe.cel_mat, rc)
    print(f'{num_rpt=}')
    max_rpt, _ = num_rpt.max(dim=0)
    sft_cel = fn.arange_prod(max_rpt * 2 + 1) - max_rpt
    sft_xyz = sft_cel.to(pe.cel_mat) @ pe.cel_mat.detach()
    _, n_pnt, _ = pe.pos_xyz.size()
    # (n_bch, n_pnt, n_pnt, n_sht, n_dim)
    pos_xyz = fn.to_unit_cell(pe.pos_xyz.detach(), pe.spc_xyz.detach())
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
        adj=adj[:, val], sft=sft_cel, spc=pe.spc_cel,
    )
    return ret


def get_minimum_radial_grid(cel_mat: Tensor, cel_rec: Tensor,
                            cut: float) -> Tensor:
    mat_rec = (cel_mat * cel_rec).sum(-1)
    mat_mat = (cel_mat * cel_mat).sum(-1)
    print(mat_rec)
    nfl = (mat_mat / (mat_rec * mat_rec) * cut * cut).sqrt()
    n = nfl.ceil().to(torch.long)
    return n


def main():
    cel, rc = corner_case_cell()
    n_atm = 10
    pos = torch.rand((1, n_atm, 3))
    pbc = torch.ones((1, 3), dtype=torch.bool)
    elm = torch.zeros((1, n_atm), dtype=torch.long)
    ent = elm >= 0
    pnt = pn.pnt_ful(cel, pbc, pos, ent)
    adj1 = coo2_ful_simple(pnt, rc)
    adj2 = coo2_ful_simple_2(pnt, rc)
    print(adj1.adj.size())
    print(adj2.adj.size())
    print((adj1.adj[:2] == adj2.adj[:2]).all())


def check(a, b, c, alpha, beta, gamma):
    rc = 20.0
    cel = triu_cell(
        CellParameter(
            a=a, b=b, c=c,
            alpha=alpha, beta=beta, gamma=gamma
        ))[None]
    cel_mat = cel
    cel_rec = cel.inverse().transpose(1, 2)
    pbc = torch.ones((1, 3), dtype=torch.bool)
    test1 = fn.minimum_neighbor(cel_rec, pbc, rc)
    test2 = get_minimum_radial_grid(cel_rec, cel_mat, rc)
    return (test1 == test2).all()


def check_all():
    x = torch.linspace(2.7, 20.0, 20)
    theta = torch.linspace(0.01, pi - 0.01, 20)
    for a in x:
        for b in x:
            for c in x:
                for alpha in theta:
                    for beta in theta:
                        for gamma in theta:
                            result = check(a, b, c, alpha, beta, gamma)
                            print(a, b, c, alpha, beta, gamma, result)
                            if not result:
                                return


def corner_case_cell():
    return triu_cell(CellParameter(
        a=2.7, b=2.7, c=20.0, alpha=0.01, beta=0.01, gamma=0.01))[None], 20.0



if __name__ == "__main__":
    main()



# どうやら、get_minimum_radial_gridを使った方がいいっぽい。
# とはいえ、ほとんどのケースでは同じ結果をもたらす。
