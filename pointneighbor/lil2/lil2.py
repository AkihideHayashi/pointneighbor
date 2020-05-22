import torch
from torch import Tensor
from ..type import AdjSftSpc, AdjSftSpcVecSod, vec_sod_adj_to_adj
from .. import functional as fn


def lil2_adj_sft_siz_vec_sod(adj_coo: AdjSftSpcVecSod):
    _, _, j, s = adj_coo.adj.unbind(0)
    mask = mask_coo_to_lil(vec_sod_adj_to_adj(adj_coo))
    ret_j = coo_to_lil(j, mask, -1)
    ret_s = coo_to_lil(s, mask, 0)
    ret_adj = torch.stack([ret_j, ret_s])
    ret_vec = coo_to_lil(adj_coo.vec, mask, 0)
    ret_sod = coo_to_lil(adj_coo.sod, mask, 0)
    return AdjSftSpcVecSod(adj=ret_adj, sft=adj_coo.sft, spc=adj_coo.spc,
                           vec=ret_vec, sod=ret_sod)


def lil2_adj_sft_siz(adj_coo: AdjSftSpc):
    _, _, j, s = adj_coo.adj.unbind(0)
    val = mask_coo_to_lil(adj_coo)
    ret_s = coo_to_lil(s, val, 0)
    ret_j = coo_to_lil(j, val, -1)
    ret = torch.stack([ret_j, ret_s])
    return AdjSftSpc(adj=ret, sft=adj_coo.sft, spc=adj_coo.spc)


def mask_coo_to_lil(adj_coo: AdjSftSpc):
    n_bch, n_pnt, _ = adj_coo.spc.size()
    nijs = adj_coo.adj
    n, i, _, _ = nijs.unbind(0)
    ni = n * n_pnt + i
    cnt = fn.count_number(ni, [n_bch * n_pnt]).view([n_bch, n_pnt])
    max_cnt = int(cnt.max().item())
    mask = fn.arange([n_bch, n_pnt, max_cnt], -1, cnt) < cnt[:, :, None]
    return mask


def coo_to_lil(coo: Tensor, mask: Tensor, dummy: int):
    size = list(mask.size())
    for s in coo.size()[1:]:
        size.append(s)
    ret = torch.ones(size, device=coo.device, dtype=coo.dtype) * dummy
    ret[mask] = coo
    return ret
