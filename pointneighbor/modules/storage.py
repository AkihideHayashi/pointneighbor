from typing import Optional
import torch
from torch import nn
from ..types import AdjSftSpc, VecSod


class AdjSftSpcStorage(nn.Module):
    def __init__(self):
        super().__init__()
        self.adj = torch.tensor([])
        self.sft = torch.tensor([])
        self.spc = torch.tensor([])

    def forward(self, adj: Optional[AdjSftSpc] = None):
        if adj is not None:
            self.adj = adj.adj
            self.sft = adj.sft
            self.spc = adj.spc
        return AdjSftSpc(adj=self.adj, sft=self.sft, spc=self.spc)

    def is_empty(self):
        if self.adj.size(0) == 0:
            assert self.sft.size(0) == 0
            assert self.spc.size(0) == 0
            return True
        else:
            assert self.sft.size(0) != 0
            assert self.spc.size(0) != 0
            return False


class VecSodStrage(nn.Module):
    def __init__(self):
        super().__init__()
        self.vec = torch.tensor([])
        self.sod = torch.tensor([])

    def forward(self, vec_sod: Optional[VecSod] = None):
        if vec_sod is not None:
            self.vec = vec_sod.vec
            self.sod = vec_sod.sod
        return VecSod(vec=self.vec, sod=self.sod)

    def is_empty(self):
        if self.vec.size(0) == 0:
            assert self.sod.size(0) == 0
            return True
        else:
            assert self.sod.size(0) != 0
            return False
