import torch
from torch import nn
from ..coo2 import (coo2_ful_simple, coo2_ful_pntsft, cel_num_div,
                    coo2_cel, cel_adj, cel_blg, CelAdj)
from ..type import (PntFul, AdjSftSpc, coo2_adj_vec_sod)


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


class CelAdjModule(nn.Module):
    def __init__(self, rc: float):
        super().__init__()
        self.adj = torch.tensor([])
        self.sft = torch.tensor([])
        self.div = torch.tensor([], dtype=torch.long)
        self.pbc = torch.tensor([], dtype=torch.bool)
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
    def __init__(self, rc: float):
        super().__init__()
        self.rc = rc
        self.cel_adj = CelAdjModule(rc)
        self.blg = torch.tensor([], dtype=torch.long)
        self.adj = torch.tensor([])
        self.sft = torch.tensor([])
        self.spc = torch.tensor([])

    def forward(self, pe: PntFul) -> AdjSftSpc:
        ca = self.cel_adj(pe)
        blg = cel_blg(ca, pe)
        if not torch.equal(blg, self.blg):
            adj = coo2_cel(ca, blg, pe.spc_cel, pe.ent)
            self.blg = blg
            self.adj = adj.adj
            self.sft = adj.sft
            self.spc = adj.spc
        adj = AdjSftSpc(adj=self.adj, sft=self.sft, spc=self.spc)
        adj, _ = coo2_adj_vec_sod(adj, pe.pos_xyz, pe.cel_mat, self.rc)
        return adj


class Coo2BookKeeping(nn.Module):
    def __init__(self, coo2, criteria, rc):
        super().__init__()
        self.coo2 = coo2
        self.criteria = criteria
        self.adj = torch.tensor([])
        self.sft = torch.tensor([])
        self.spc = torch.tensor([])
        self.rc = rc

    def forward(self, pe: PntFul):
        if self.criteria(pe):
            adj: AdjSftSpc = self.coo2(pe)
            self.adj = adj.adj
            self.sft = adj.sft
            self.spc = adj.spc
        adj = AdjSftSpc(adj=self.adj, sft=self.sft, spc=self.spc)
        adj, _ = coo2_adj_vec_sod(adj, pe.pos_xyz, pe.cel_mat, self.rc)
        return adj


class VerletCriteria(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.i = 0

    def forward(self, _: PntFul):
        ret = self.i % self.n == 0
        self.i += 1
        return ret


class StrictCriteria(nn.Module):
    def __init__(self, delta, debug: bool = False):
        super().__init__()
        self.delta = delta
        self.do2 = delta / 2
        self.pos_xyz = torch.tensor([])
        self.cel_mat = torch.tensor([])
        self.debug = debug

    def forward(self, pnt: PntFul):
        if self._criteria(pnt):
            self.pos_xyz = pnt.pos_xyz
            self.cel_mat = pnt.cel_mat
            if self.debug:
                print('StrictCriteria: calc')
            return True
        else:
            if self.debug:
                print('StrictCriteria: skip')
            return False

    def _criteria(self, pnt: PntFul):
        cel_mat = pnt.cel_mat
        pos_xyz = pnt.pos_xyz
        if cel_mat.size() != self.cel_mat.size():
            return True
        if pos_xyz.size() != self.pos_xyz.size():
            return True
        if not torch.equal(cel_mat, self.cel_mat):
            return True
        if (self.pos_xyz - pos_xyz).pow(2).sum(-1).max() >= self.do2:
            return True
        return False
