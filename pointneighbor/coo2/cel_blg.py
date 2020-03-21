import torch
from torch import Tensor
from .. import functional as fn
from ..type import PntExp
from .cel_adj import CelAdj


def cel_blg(cel_adj: CelAdj, pe: PntExp) -> Tensor:
    assert fn.in_unit_cell(pe.pos_cel, pe.ent)
    if not fn.in_unit_cell(pe.pos_cel, pe.ent):
        raise RuntimeError("cell index supports points not in cell.")
    num_div = cel_adj.div[:, None, :]
    parcel = (num_div * pe.pos_cel).floor().to(torch.int64)
    return fn.ravel(parcel, num_div, -1)
