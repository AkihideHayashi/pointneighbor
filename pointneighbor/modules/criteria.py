"""Provides a criterion for whether or not a point has "moved enough"."""
import torch
from torch import nn, Tensor
from ..types import PntFul


def _moved(a, b):
    if a.size() != b.size():
        return True
    else:
        return bool((a != b).any().item())


class MovedCriteria(nn.Module):
    cel: Tensor
    pos: Tensor
    pbc: Tensor
    ent: Tensor

    def __init__(self):
        super().__init__()
        self.cel = torch.tensor([])
        self.pos = torch.tensor([])
        self.pbc = torch.tensor([])
        self.ent = torch.tensor([])

    def forward(self, pf: PntFul):
        ret = self._criteria(pf)
        if ret:
            self.cel = pf.cel_mat
            self.pos = pf.pos_xyz
            self.pbc = pf.pbc
            self.ent = pf.ent
        return ret

    def _criteria(self, pf: PntFul):
        return (_moved(self.pbc, pf.pbc) or _moved(self.cel, pf.cel_mat) or
                _moved(self.ent, pf.ent) or _moved(self.pos, pf.pos_xyz))


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
        self.do2 = delta * delta / 4
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
