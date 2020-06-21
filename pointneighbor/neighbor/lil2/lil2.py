import torch
from torch import Tensor
from ...types import AdjSftSpc, VecSod
from ... import functional as fn


def coo_to_lil(adj_coo: AdjSftSpc):
    """Transform adjacent information coo to lil.

    Args:
        adj_coo (AdjSftSpc): adjacent information in coo format.

    Returns:
        adjacent information in lil format.
    """
    _, _, j, s = adj_coo.adj.unbind(0)
    val = transformation_mask_coo_to_lil(adj_coo)
    ret_s = transform_tensor_coo_to_lil(s, val, 0)
    ret_j = transform_tensor_coo_to_lil(j, val, -1)
    ret = torch.stack([ret_j, ret_s])
    return AdjSftSpc(adj=ret, sft=adj_coo.sft, spc=adj_coo.spc)


def transformation_mask_coo_to_lil(adj_coo: AdjSftSpc):
    """Mask to transform coo to lil."""
    n_bch, n_pnt, _ = adj_coo.spc.size()
    nijs = adj_coo.adj
    n, i, _, _ = nijs.unbind(0)
    ni = n * n_pnt + i
    cnt = fn.count_number(ni, [n_bch * n_pnt]).view([n_bch, n_pnt])
    max_cnt = int(cnt.max().item())
    mask = fn.arange([n_bch, n_pnt, max_cnt], -1, cnt) < cnt[:, :, None]
    return mask


def transform_tensor_coo_to_lil(coo: Tensor, mask: Tensor, dummy: int):
    """Transform arbitrary 2-body tensor. coo2 -> lil2

    Args:
        coo: The tensor to transform.
        mask: The result of transformation_mask_coo_to_lil

    """
    size = list(mask.size())
    for s in coo.size()[1:]:
        size.append(s)
    ret = torch.ones(size, device=coo.device, dtype=coo.dtype) * dummy
    ret[mask] = coo
    return ret


def __coo2_to_lil_adj_sft_spc_vec_sod(adj_coo: AdjSftSpc, vec_sod_coo: VecSod):
    _, _, j, s = adj_coo.adj.unbind(0)
    mask = transformation_mask_coo_to_lil(adj_coo)
    ret_j = transform_tensor_coo_to_lil(j, mask, -1)
    ret_s = transform_tensor_coo_to_lil(s, mask, 0)
    ret_adj = torch.stack([ret_j, ret_s])
    ret_vec = transform_tensor_coo_to_lil(vec_sod_coo.vec, mask, 0)
    ret_sod = transform_tensor_coo_to_lil(vec_sod_coo.sod, mask, 0)
    ret1 = AdjSftSpc(adj=ret_adj, sft=adj_coo.sft, spc=adj_coo.spc)
    ret2 = VecSod(vec=ret_vec, sod=ret_sod)
    return ret1, ret2
