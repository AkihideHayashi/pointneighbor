from torch import Tensor
import torch
from .... import functional as fn
from ....types import AdjSftSpc
from .cel_adj import CelAdj


def _sort_coo2(adj_coo: AdjSftSpc):
    n, i, j, s = adj_coo.adj.unbind(0)
    i_max = i.max() + 5
    j_max = j.max() + 5
    s_max = s.max() + 5
    x = ((n * i_max + i) * j_max + j) * s_max + s
    _, idx = x.sort()
    return AdjSftSpc(adj_coo.adj[:, idx], adj_coo.sft, adj_coo.spc)


def coo2_cel(cel_adj: CelAdj, blg: Tensor,
             spc: Tensor, ent: Tensor) -> AdjSftSpc:
    """Main code which generate adjacent
    Args:
        cel_adj (CelAdj): Cell adjacent information. Result of cel_adj
        blg (int[bch, pnt]): Belong information. Result of cel_blg.
        spc (int[bch, pnt, dim]): Which unit lattice points belong.
        ent (bool[bch, pnt]): not dummy.
    """
    pnt_pcl = _pnt_pcl(cel_adj, blg)  # n_bch, n_pcl, n_cnt
    _, _, n_cnt = pnt_pcl.size()
    nn, ii, jj, ss = cel_adj.adj.unbind(0)
    n_adj, = nn.size()
    size = [n_adj, n_cnt, n_cnt]
    n = nn[:, None, None].expand(size)
    i = pnt_pcl[nn, ii, :][:, :, None].expand(size)
    j = pnt_pcl[nn, jj, :][:, None, :].expand(size)
    s = ss[:, None, None].expand(size)
    adj = _contraction(n, i, j, s, cel_adj.sft, ent)
    return _sort_coo2(AdjSftSpc(adj=adj, sft=cel_adj.sft, spc=spc))


def _pnt_pcl(cel_adj: CelAdj, blg: Tensor) -> Tensor:
    """
    pnt_pcl[i_bch, i_pcl, i_per] is i_per'th atom's index
    in i_pcl'th percel in i_bch'th batch.
    """
    blg_srt, idx = blg.sort()
    n_pcl = int(cel_adj.div.prod(dim=1).max().item())
    n_bch, _ = blg.size()
    cnt = torch.zeros([n_bch, n_pcl]).to(blg)
    cnt.scatter_add_(dim=1, index=blg_srt, src=torch.ones_like(blg_srt))
    max_cnt = int(cnt.max().item())
    cnt_exp = cnt[:, :, None].expand((-1, -1, max_cnt))
    val = fn.arange_like(cnt_exp, dim=-1) < cnt_exp
    size = [n_bch, n_pcl, max_cnt]
    ret = torch.ones(size).to(blg) * -1
    ret[val] = idx.flatten(0, 1)
    return ret


def _contraction(n, i, j, s, sft, ent):
    ei = ent[n, i]
    ej = ent[n, j]
    val_ent = ei & ej
    val_dum = (i >= 0) & (j >= 0)
    val_idt = (sft != 0).any(dim=-1)[s] | (i != j)
    val = val_dum & val_idt & val_ent
    return torch.stack([n[val], i[val], j[val], s[val]])
