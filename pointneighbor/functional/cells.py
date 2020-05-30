
import torch
from torch import Tensor


def get_pos_spc(pos_cel: Tensor, pbc: Tensor):
    return pos_cel.floor() * pbc[:, None, :]


def to_unit_cell(pos: Tensor, spc: Tensor):
    return pos - spc


def in_unit_cell(pos_cel: Tensor, ent: Tensor):
    """0 <= pos_ltc <= 1"""
    in_ = ((pos_cel >= -1.0e-8) & (pos_cel <= 1 + 1.0e-8))
    in_[~ent] = torch.tensor(True)
    return in_.all()


def minimum_neighbor(cel_rec: Tensor, pbc: Tensor, rc: float) -> Tensor:
    """Minimum number of repeats required by the rc."""
    max_repeats = cel_rec.detach().norm(p=2, dim=-1) * rc
    repeats = max_repeats.ceil().to(torch.int64)
    return repeats * pbc.type_as(repeats)
