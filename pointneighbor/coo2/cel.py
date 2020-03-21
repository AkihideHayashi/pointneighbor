from torch import Tensor
import torch
from .. import functional as fn
from ..type import Adj
from .cel_adj import CelAdj


def _pnt_pcl(cel_adj: CelAdj, blg: Tensor):
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


def coo2_cel(cel_adj: CelAdj, blg: Tensor) -> Adj:
    pnt_pcl = _pnt_pcl(cel_adj, blg)  # n_bch, n_pcl, n_cnt
    _, _, n_cnt = pnt_pcl.size()
    nn, ii, jj, ss = cel_adj.adj.unbind(0)
    n_adj, = nn.size()
    size = [n_adj, n_cnt, n_cnt]
    n = nn[:, None, None].expand(size)
    i = pnt_pcl[nn, ii, :][:, :, None].expand(size)
    j = pnt_pcl[nn, jj, :][:, None, :].expand(size)
    s = ss[:, None, None].expand(size)
    adj = _contraction(n, i, j, s, cel_adj.sft)
    return Adj(adj=adj, sft=cel_adj.sft)


def _contraction(n, i, j, s, sft):
    val_dum = (i >= 0) & (j >= 0)
    val_idt = (sft != 0).any(dim=-1)[s] | (i != j)
    val = val_dum & val_idt
    return torch.stack([n[val], i[val], j[val], s[val]])
