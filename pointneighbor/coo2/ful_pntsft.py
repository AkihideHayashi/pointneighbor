import torch
from torch import Tensor
from ..type import PntExp, VecSodAdj
from .. import functional as fn


# (n_bch, n_pnt, n_pnt x n_sft - delta, n_dim)


def coo2_ful_pntsft(pe: PntExp, rc: float) -> VecSodAdj:
    """An implementation for make coo-like 2-body problem.
    Make (n_bch, n_pnt, n_pnt x n_sft - delta, n_dim)
    tensor and remove redundants.
    Middle overhead and middle efficient in terms of computational complexity.
    Use this function when your system is small and
    the size of lattice is not uniform.
    This function is not efficient if your system is big.
    If lattice is uniform, coo2_fullindex_simple has less overhead.
    """
    device = pe.pos_xyz.device
    n_bch, n_pnt, _ = pe.pos_xyz.size()
    num_rpt = fn.minimum_neighbor(pe.cel_rec, pe.pbc, rc)
    max_rpt, _ = num_rpt.max(dim=0)
    sft_cel = fn.arange_prod(max_rpt * 2 + 1) - max_rpt
    sft_xyz = sft_cel.to(pe.cel_mat) @ pe.cel_mat
    n_sft, _ = sft_cel.size()
    pos_i = pe.pos_xyz
    pos_j = pe.pos_xyz[:, :, None, :] + sft_xyz[:, None, :, :]

    msk_f, msk_t = _mask_transform(sft_cel, num_rpt, pe.pbc, n_bch, n_pnt)
    pos_flt_j = _transform(pos_j, msk_f, msk_t)[:, None, :, :]
    pos_flt_i = pos_i[:, :, None, :]
    vec = pos_flt_i - pos_flt_j
    sod = fn.square_of_distance(vec, dim=-1)
    val_cut = sod <= rc * rc
    ent_i = pe.ent[:, :, None]
    val = val_cut & msk_t[:, None, :] & ent_i
    n = fn.arange([n_bch, n_pnt], dim=0).to(device)[:, :, None].expand_as(sod)
    i = fn.arange([n_bch, n_pnt], dim=1).to(device)[:, :, None].expand_as(sod)
    j = _transform(fn.arange([n_bch, n_pnt, n_sft], dim=1).to(device),
                   msk_f, msk_t)[:, None, :].expand_as(sod)
    s = _transform(fn.arange([n_bch, n_pnt, n_sft], dim=2).to(device),
                   msk_f, msk_t)[:, None, :].expand_as(sod)
    adj = torch.stack([n[val], i[val], j[val], s[val]])
    vsa = VecSodAdj(vec=vec[val], sod=sod[val], adj=adj, sft=sft_cel)
    ret = _contract_idt_ent(vsa, pe.ent)
    return ret


def _contract_idt_ent(vsa: VecSodAdj, ent: Tensor):
    vec, sod, adj, sft = vsa
    n, i, j, s = adj.unbind(0)
    z_sft = (sft == 0).all(dim=-1)
    val_idt = ~z_sft[s] | (i != j)
    ej = ent[n, j]
    val_ent = ej
    val = val_idt & val_ent
    return VecSodAdj(vec=vec[val], sod=sod[val], adj=adj[:, val], sft=sft)


def _transform(tensor: Tensor, val_f: Tensor, val_t: Tensor):
    size = list(val_t.size()) + list(tensor.size())[3:]
    ret = torch.zeros(size).to(tensor)
    ret[val_t] = tensor[val_f]
    return ret


def _mask_transform(sft_cel, num_rpt, pbc, n_bch: int, n_pnt: int):
    msk_f = _make_msk_pnt_sft(sft_cel, num_rpt, pbc, n_pnt)
    msk_t = _make_msk_pntsft(msk_f, n_bch)
    return msk_f, msk_t


def _make_msk_pntsft(val_f, n_bch: int):
    n = fn.arange_like(val_f, dim=0)[val_f]
    count = fn.count_number(n, [n_bch])
    return torch.arange(count.max()).to(count)[None, :] < count[:, None]


def _make_msk_pnt_sft(sft_cel, num_rpt, pbc, n_pnt: int):
    val_pbc = fn.val_pbc(sft_cel, pbc)
    val_sft = (sft_cel.abs()[None, :, :] <= num_rpt[:, None, :]).all(dim=-1)
    val = (val_pbc & val_sft)[:, None, :].expand((-1, n_pnt, -1))
    return val
