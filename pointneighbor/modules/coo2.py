import torch
from torch import nn
from .. import functional as fn
from ..coo2 import (coo2_ful_simple, coo2_ful_pntsft,
                    coo2_cel, cel_adj, cel_blg, CelAdj)
from ..type import PntExp, VecSodAdj, vec_sod_adj, contract, Adj


class Coo2FulSimple(nn.Module):
    def __init__(self, rc: float):
        super().__init__()
        self.rc = rc

    def forward(self, pe: PntExp) -> VecSodAdj:
        return coo2_ful_simple(pe, self.rc)


class Coo2FulPntSft(nn.Module):
    def __init__(self, rc: float):
        super().__init__()
        self.rc = rc

    def forward(self, pe: PntExp) -> VecSodAdj:
        return coo2_ful_pntsft(pe, self.rc)


class CelAdjModule(nn.Module):
    def __init__(self, rc: float):
        super().__init__()
        self.adj = torch.tensor([])
        self.sft = torch.tensor([])
        self.div = torch.tensor([])
        self.rc = rc
        self.cel_mat = torch.tensor([])
        self.pbc = torch.tensor([], dtype=torch.bool)

    def forward(self, pe: PntExp):
        if self.immute(pe):
            return self.cell_adj()
        ca = cel_adj(pe, self.rc)
        self.register(ca)
        return self.cell_adj()

    def register(self, ca: CelAdj):
        self.adj = ca.adj
        self.sft = ca.sft
        self.div = ca.div

    def cell_adj(self):
        return CelAdj(adj=self.adj, sft=self.sft, div=self.div)

    def immute(self, pe: PntExp):
        if not torch.equal(self.cel_mat, pe.cel_mat):
            return False
        if not torch.equal(self.pbc, pe.pbc):
            return False
        return True


class Coo2Cel(nn.Module):
    def __init__(self, rc: float):
        super().__init__()
        self.rc = rc
        self.cel_adj = CelAdjModule(rc)
        self.blg = torch.tensor([], dtype=torch.long)
        self.adj = torch.tensor([])
        self.sft = torch.tensor([])

    def forward(self, pe: PntExp) -> VecSodAdj:
        ca = self.cel_adj(pe)
        blg = cel_blg(ca, pe)
        if torch.equal(blg, self.blg):
            return self.vsa(pe)
        adj = coo2_cel(ca, blg)
        self.blg = blg
        self.adj = adj.adj
        self.sft = adj.sft
        return self.vsa(pe)

    def vsa(self, pe: PntExp) -> VecSodAdj:
        adj = Adj(adj=self.adj, sft=self.sft)
        vsa = vec_sod_adj(pe, adj, self.rc)
        return contract(vsa, pe, self.rc)


class Coo2BookKeeping(nn.Module):
    def __init__(self, coo2, rc: float, delta: float):
        super().__init__()
        self.coo2 = coo2(rc + delta)
        self.rc = rc
        self.delta = delta
        self.pos_xyz = torch.tensor([])
        self.pos_cel = torch.tensor([])
        self.cel_mat = torch.tensor([])
        self.adj = torch.tensor([])
        self.sft = torch.tensor([])

    def forward(self, pe: PntExp):
        if self.immute(pe):
            return self.coo(pe)
        self.register(pe)
        return self.coo(pe)

    def coo(self, pe: PntExp):
        adj = Adj(self.adj, self.sft)
        return contract(vec_sod_adj(pe, adj, self.rc), pe, self.rc)

    def register(self, pe: PntExp):
        adj = self.coo2(pe)
        self.adj = adj.adj
        self.sft = adj.sft
        self.cel_mat = pe.cel_mat
        self.pos_xyz = pe.pos_xyz
        self.pos_cel = pe.pos_cel

    def immute(self, pe: PntExp):
        if not torch.equal(pe.cel_mat, self.cel_mat):
            return False
        if self.move_pos(pe):
            return False
        return True

    def move_pos(self, pe: PntExp):
        n, i, j, s = self.adj.unbind(0)
        num_rep, _ = self.sft.max(dim=0)

        sft_exp_cel = (pe.pos_cel - self.pos_cel).round().to(torch.long)
        sft_exp_xyz = sft_exp_cel.to(self.cel_mat) @ self.cel_mat

        ss = self.sft[s]
        ss += sft_exp_cel[n, i]
        ss -= sft_exp_cel[n, j]
        s = fn.ravel1(ss + num_rep[None, :], num_rep * 2 + 1, -1)
        self.adj[-1, :] = s
        self.pos_cel += sft_exp_cel
        self.pos_xyz += sft_exp_xyz

        tol = self.delta * self.delta / 4
        vec = pe.pos_xyz - self.pos_xyz
        sod = vec.pow(2).sum(-1)
        return sod.max() >= tol
