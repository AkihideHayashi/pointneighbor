import torch
from .... import functional as fn
from ....types import PntFul
from .cel_adj import CelAdj


def cel_blg(cel_adj: CelAdj, pe: PntFul):
    """which parcel the point belongs."""
    pos_cel = fn.to_unit_cell(pe.pos_cel, pe.spc_cel)
    num_div = cel_adj.div[:, None, :]
    parcel = (num_div * pos_cel).floor().to(torch.int64)
    mask = parcel < num_div
    assert (parcel[~mask] == num_div.expand_as(mask)[~mask]).all()
    parcel = torch.where(mask, parcel, num_div - 1)
    blg = fn.ravel(parcel, num_div, -1)
    return blg
