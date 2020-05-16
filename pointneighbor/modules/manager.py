import torch
from torch import nn
from ..type import Pnt, pnt_exp, PntExp, AdjSftSizVecSod


class PntModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_xyz = torch.tensor([])
        self.pos_cel = torch.tensor([])
        self.cel_mat = torch.tensor([])
        self.cel_inv = torch.tensor([])
        self.cel_rec = torch.tensor([])
        self.ent = torch.tensor([], dtype=torch.bool)
        self.pbc = torch.tensor([], dtype=torch.bool)

    def set(self, p: Pnt):
        pe = pnt_exp(p)
        self.cel_mat = pe.cel_mat
        self.cel_inv = pe.cel_inv
        self.cel_rec = pe.cel_rec
        self.pbc = pe.pbc
        self.pos_cel = pe.pos_cel
        self.pos_xyz = pe.pos_xyz
        self.ent = pe.ent

    def get(self):
        return PntExp(
            cel_mat=self.cel_mat, cel_inv=self.cel_inv, cel_rec=self.cel_rec,
            pbc=self.pbc, pos_cel=self.pos_cel, pos_xyz=self.pos_xyz,
            ent=self.ent
        )

    def forward(self, p: Pnt):
        self.set(p)
        return self.get()


class Coo2AdjSftSizVecSod(nn.Module):
    def __init__(self, coo2):
        """
        Args:
            coo2: instane of Union[Coo2FulSimple, Coo2FulPntSft,
                                   Coo2Cel, Coo2BookKeeping]
        """
        super().__init__()
        self.pnt = PntModule()
        self.coo2 = coo2
        self.vec = torch.tensor([])
        self.sod = torch.tensor([])
        self.adj = torch.tensor([], dtype=torch.long)
        self.sft = torch.tensor([], dtype=torch.long)

    def set(self, p: Pnt):
        pe: PntExp = self.pnt(p)
        asvs: AdjSftSizVecSod = self.coo2(pe)
        self.vec = asvs.vec
        self.sod = asvs.sod
        self.adj = asvs.adj
        self.sft = asvs.sft

    def get(self):
        return AdjSftSizVecSod(vec=self.vec, sod=self.sod,
                               adj=self.adj, sft=self.sft)

    def forward(self, p: Pnt):
        self.set(p)
        return self.get()


class Coo2AdjSftSizVecSodManager(nn.Module):
    def __init__(self, coo2):
        super().__init__()
        self.coo2 = Coo2AdjSftSizVecSod(coo2)
