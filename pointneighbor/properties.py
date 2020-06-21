import torch
from .types import AdjSftSpc, is_lil2, is_coo2
from . import functional as fn


def coo2_n_i_j(adj: AdjSftSpc):
    assert is_coo2(adj)
    n, i, j, _ = adj.adj.unbind(0)
    return n, i, j


def coo2_n_i_j_sft(adj: AdjSftSpc):
    assert is_coo2(adj)
    n, i, j, s = adj.adj.unbind(0)
    spc = fn.to_unit_cell(torch.zeros_like(adj.spc), adj.spc)
    spc_i = spc[n, i]
    spc_j = spc[n, j]
    sft = fn.vector(spc_i, spc_j, adj.sft[s])
    return n, i, j, sft


def lil2_j_s(adj: AdjSftSpc):
    assert is_lil2(adj)
    j, s = adj.adj.unbind(0)
    assert j.dim() == 3  # n_bch, n_pnt, n_cnt
    i = fn.arange_like(j, 1)
    n = fn.arange_like(j, 0)
    spc = fn.to_unit_cell(torch.zeros_like(adj.spc), adj.spc)
    spc_i = spc[n, i]
    spc_j = spc[n, j]
    sft = fn.vector(spc_i, spc_j, adj.sft[s])
    return j, sft
