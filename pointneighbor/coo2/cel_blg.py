import torch
from .. import functional as fn
from ..type import PntExt
from .cel_adj import CelAdj


def cel_blg(cel_adj: CelAdj, ep: PntExt):
    num_div = cel_adj.div[:, None, :]
    parcel = (num_div * ep.pos_cel).floor().to(torch.int64)
    return fn.ravel(parcel, num_div, -1)
