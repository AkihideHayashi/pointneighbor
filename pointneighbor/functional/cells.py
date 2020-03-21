
import torch
from torch import Tensor


def to_unit_cell(pos_ltc: Tensor, pbc: Tensor):
    """Mod pos_ltc 1."""
    sft = -pos_ltc.floor() * pbc[:, None, :]
    return pos_ltc + sft


def in_unit_cell(pos_ltc: Tensor, ent: Tensor):
    """0 <= pos_ltc <= 1"""
    in_ = ((pos_ltc >= -1.0e-8) & (pos_ltc <= 1 + 1.0e-8))
    in_[~ent] = torch.tensor(True)
    return in_.all()


def minimum_neighbor(ltc_rec: Tensor, pbc: Tensor, rc: float) -> Tensor:
    """Minimum number of repeats required by the rc."""
    max_repeats = ltc_rec.detach().norm(p=2, dim=-1) * rc
    repeats = max_repeats.ceil().to(torch.int64)
    return repeats * pbc.type_as(repeats)
