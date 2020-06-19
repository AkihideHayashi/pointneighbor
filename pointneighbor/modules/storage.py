from typing import Optional
import torch
from torch import nn
from ..type import AdjSftSpc


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

    def empty(self):
        if self.adj.size(0) == 0:
            assert self.sft.size(0) == 0
            assert self.spc.size(0) == 0
            return True
        else:
            assert self.sft.size(0) != 0
            assert self.spc.size(0) != 0
            return False
