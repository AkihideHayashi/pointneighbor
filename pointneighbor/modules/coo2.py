import torch
from torch import nn, Tensor
from ..neighbor.coo2 import (coo2_ful_simple, coo2_ful_pntsft, cel_num_div,
                             coo2_cel, cel_adj, cel_blg, CelAdj)
from ..types import PntFul, AdjSftSpc
from ..utils import cutoff_coo2, coo2_vec_sod
from .storage import AdjSftSpcStorage


class Coo2FulSimple(nn.Module):
    def __init__(self, rc: float):
        super().__init__()
        self.rc = rc

    def forward(self, pe: PntFul) -> AdjSftSpc:
        return coo2_ful_simple(pe, self.rc)


class Coo2FulPntSft(nn.Module):
    def __init__(self, rc: float):
        super().__init__()
        self.rc = rc

    def forward(self, pe: PntFul) -> AdjSftSpc:
        return coo2_ful_pntsft(pe, self.rc)


class _CelAdjModule(nn.Module):
    adj: Tensor
    sft: Tensor
    div: Tensor
    pbc: Tensor

    def __init__(self, rc: float):
        super().__init__()
        self.register_buffer('adj', torch.tensor([]))
        self.register_buffer('sft', torch.tensor([]))
        self.register_buffer('div', torch.tensor([], dtype=torch.long))
        self.register_buffer('pbc', torch.tensor([], dtype=torch.bool))
        self.rc = rc

    def forward(self, pe: PntFul) -> CelAdj:
        if self._moved(pe):
            ca = cel_adj(pe, self.rc)
            self.adj = ca.adj
            self.sft = ca.sft
            self.div = ca.div
            self.pbc = pe.pbc
        return CelAdj(adj=self.adj, sft=self.sft, div=self.div)

    def _moved(self, pe: PntFul):
        div = cel_num_div(pe.cel_mat, self.rc)
        if not torch.equal(self.div, div):
            return True
        if not torch.equal(self.pbc, pe.pbc):
            return True
        return False


class Coo2Cel(nn.Module):
    blg: Tensor

    def __init__(self, rc: float):
        super().__init__()
        self.rc = rc
        self.cel_adj = _CelAdjModule(rc)
        self.register_buffer('blg', torch.tensor([], dtype=torch.long))
        self.adj = AdjSftSpcStorage()

    def forward(self, pe: PntFul) -> AdjSftSpc:
        ca = self.cel_adj(pe)
        blg = cel_blg(ca, pe)
        if not torch.equal(blg, self.blg):
            adj = coo2_cel(ca, blg, pe.spc_cel, pe.ent)
            self.blg = blg
            self.adj(adj)
        adj = self.adj()
        vec_sod = coo2_vec_sod(adj, pe.pos_xyz, pe.cel_mat)
        adj_cut, _ = cutoff_coo2(adj, vec_sod, self.rc)
        return adj_cut


class Coo2BookKeeping(nn.Module):
    adj: Tensor
    sft: Tensor
    spc: Tensor

    def __init__(self, coo2, criteria, rc):
        super().__init__()
        self.coo2 = coo2
        self.criteria = criteria
        self.register_buffer('adj', torch.tensor([]))
        self.register_buffer('sft', torch.tensor([]))
        self.register_buffer('spc', torch.tensor([]))
        self.rc = rc

    def forward(self, pe: PntFul):
        if self.criteria(pe):
            adj: AdjSftSpc = self.coo2(pe)
            self.adj = adj.adj
            self.sft = adj.sft
            self.spc = adj.spc
        adj = AdjSftSpc(adj=self.adj, sft=self.sft, spc=self.spc)
        vec_sod = coo2_vec_sod(adj, pe.pos_xyz, pe.cel_mat)
        adj, _ = cutoff_coo2(adj, vec_sod, self.rc)
        return adj
