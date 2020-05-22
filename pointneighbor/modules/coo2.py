import torch
from torch import nn
from .. import functional as fn
from ..coo2 import (coo2_ful_simple, coo2_ful_pntsft, cel_num_div,
                    coo2_cel, cel_adj, cel_blg, CelAdj)
from ..type import (PntExp, AdjSftSpcVecSod, vec_sod_adj, contract, AdjSftSpc)


class Coo2FulSimple(nn.Module):
    def __init__(self, rc: float):
        super().__init__()
        self.rc = rc

    def forward(self, pe: PntExp) -> AdjSftSpcVecSod:
        adj = coo2_ful_simple(pe, self.rc)
        vsa = vec_sod_adj(pe, adj, self.rc)
        return vsa


class Coo2FulPntSft(nn.Module):
    def __init__(self, rc: float):
        super().__init__()
        self.rc = rc

    def forward(self, pe: PntExp) -> AdjSftSpcVecSod:
        adj = coo2_ful_pntsft(pe, self.rc)
        vsa = vec_sod_adj(pe, adj, self.rc)
        return vsa


class CelAdjModule(nn.Module):
    def __init__(self, rc: float):
        super().__init__()
        self.adj = torch.tensor([])
        self.sft = torch.tensor([])
        self.div = torch.tensor([], dtype=torch.long)
        self.rc = rc
        self.cel_mat = torch.tensor([])
        self.pbc = torch.tensor([], dtype=torch.bool)

    def forward(self, pe: PntExp):
        if self.moved(pe):
            ca = cel_adj(pe, self.rc)
            self.register(ca)
        return self.cell_adj()

    def register(self, ca: CelAdj):
        self.adj = ca.adj
        self.sft = ca.sft
        self.div = ca.div

    def cell_adj(self):
        return CelAdj(adj=self.adj, sft=self.sft, div=self.div)

    def moved(self, pe: PntExp):
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

    def forward(self, pe: PntExp) -> AdjSftSpcVecSod:
        ca = self.cel_adj(pe)
        blg, sft = cel_blg(ca, pe)
        if torch.equal(blg, self.blg):
            return self.vsa(pe)
        adj = coo2_cel(ca, blg, sft)
        self.blg = blg
        self.adj = adj.adj
        self.sft = adj.sft
        self.spc = adj.spc
        return self.vsa(pe)

    def vsa(self, pe: PntExp) -> AdjSftSpcVecSod:
        adj = AdjSftSpc(adj=self.adj, sft=self.sft, spc=self.spc)
        vsa = vec_sod_adj(pe, adj, self.rc)
        return contract(vsa, pe, self.rc)


class Coo2BookKeeping(nn.Module):
    def __init__(self, coo2, rc: float, delta: float, debug: bool = False):
        super().__init__()
        self.coo2 = coo2(rc + delta)
        self.rc = rc
        self.delta = delta
        self.pos_xyz = torch.tensor([])
        self.pos_cel = torch.tensor([])
        self.cel_mat = torch.tensor([])
        self.adj = torch.tensor([])
        self.sft = torch.tensor([])
        self.spc = torch.tensor([])
        self.debug = debug

    def forward(self, pe: PntExp):
        if not self.immute(pe):
            if self.debug:
                print('Coo2BookKeeping: calc')
            self.register(pe)
        else:
            if self.debug:
                print('Coo2BookKeeping: skip')
        return self.coo(pe)

    def coo(self, pe: PntExp):
        adj = AdjSftSpc(adj=self.adj, sft=self.sft, spc=self.spc)
        return contract(vec_sod_adj(pe, adj, self.rc), pe, self.rc)

    def register(self, pe: PntExp):
        adj = self.coo2(pe)
        self.adj = adj.adj
        self.sft = adj.sft
        self.spc = adj.spc
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

        ss = self.sft.clone()[s]
        ss += sft_exp_cel[n, i] * pe.pbc[n, :]
        ss -= sft_exp_cel[n, j] * pe.pbc[n, :]
        s = fn.ravel1(ss + num_rep[None, :], num_rep * 2 + 1, -1)
        self.adj = torch.stack([n, i, j, s])
        self.pos_cel += sft_exp_cel
        self.pos_xyz += sft_exp_xyz

        tol = self.delta * self.delta / 4
        vec = pe.pos_xyz - self.pos_xyz
        sod = vec.pow(2).sum(-1)
        return sod.max() >= tol
