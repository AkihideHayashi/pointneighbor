import torch
from .. import functional as fn
from ..type import PntExp
from .cel_adj import CelAdj


def cel_blg(cel_adj: CelAdj, pe: PntExp):
    pos_cel = fn.to_unit_cell(pe.pos_cel, pe.sft_cel)
    num_div = cel_adj.div[:, None, :]
    parcel = (num_div * pos_cel).floor().to(torch.int64)
    return fn.ravel(parcel, num_div, -1)
