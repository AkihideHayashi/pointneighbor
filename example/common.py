import math
from typing import List, NamedTuple
from torch import Tensor
import torch
from ase import Atoms
from torchani import aev


class Pnt(NamedTuple):
    cel: Tensor
    pbc: Tensor
    pos: Tensor
    ent: Tensor


class CellParameter(NamedTuple):
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float


def triu_cell(param: CellParameter) -> Tensor:
    a, b, c, alpha, beta, gamma = param
    c3 = c
    b3 = alpha / c3
    b2 = math.sqrt(b * b - b3 * b3)
    a3 = beta / c3
    a2 = (gamma - a3 * b3) / b2
    a1 = math.sqrt(a * a - a2 * a2 - a3 * a3)
    return torch.tensor([[a1, a2, a3],
                         [0, b2, b3],
                         [0, 0, c3]])


def cell_parameter(cell: Tensor) -> CellParameter:
    idx = torch.triu_indices(3, 3)
    LL = (cell @ cell.T)
    aa, gamma, beta, bb, alpha, cc = LL[idx.unbind(0)]
    return CellParameter(
        aa.sqrt().item(),
        bb.sqrt().item(),
        cc.sqrt().item(),
        alpha.item(),
        beta.item(),
        gamma.item()
    )


def n_duo_single_lattice(p: Pnt, rc: float):
    cutoff = rc
    cell = p.cel[0]
    pbc = p.pbc[0]
    padding_mask = ~p.ent
    coordinates = p.pos
    shifts = aev.compute_shifts(cell, pbc, cutoff)
    i, _, _ = aev.neighbor_pairs(
        padding_mask, coordinates, cell, shifts, cutoff)
    return len(i) * 2


def n_trio_single_lattice(p: Pnt, rc: float):
    cutoff = rc
    cell = p.cel[0]
    pbc = p.pbc[0]
    padding_mask = ~p.ent
    coordinates = p.pos
    shifts = aev.compute_shifts(cell, pbc, cutoff)
    i, j, _ = aev.neighbor_pairs(
        padding_mask, coordinates, cell, shifts, cutoff)
    a3, _, _, _, _ = aev.triple_by_molecule(i, j)
    return len(a3) * 2


def n_duo_multi_lattice(p: Pnt, rc: float):
    n_bch = p.cel.size()[0]
    n = 0
    for i_bch in range(n_bch):
        cutoff = rc
        cell = p.cel[i_bch]
        pbc = p.pbc[i_bch]
        padding_mask = ~p.ent[i_bch][None]
        coordinates = p.pos[i_bch][None]
        shifts = aev.compute_shifts(cell, pbc, cutoff)
        i, _, _ = aev.neighbor_pairs(
            padding_mask, coordinates, cell, shifts, cutoff)
        n += len(i) * 2
    return n


def random_particle(n_bch: int, n_pnt: int, n_dim: int,
                    params: List[CellParameter], pbc: Tensor):
    cel = torch.stack([triu_cell(param) for param in params])
    pos = torch.rand((n_bch, n_pnt, n_dim)) @ cel
    ent = torch.randint(-1, 3, (n_bch, n_pnt)) >= 0
    p = Pnt(cel, pbc, pos, ent)
    return p


def ase_to_particles(mols: List[Atoms]):
    positions = [torch.tensor(atoms.positions).to(torch.float)
                 for atoms in mols]
    n = torch.tensor([pos.shape[0] for pos in positions])
    valid = torch.arange(n.max().item())[None, :] < n[:, None]
    pbc = torch.tensor([atoms.pbc.tolist() for atoms in mols])
    pos = fill(positions, valid)
    ent = fill([torch.ones(int(ni.item())) for ni in n], valid).to(torch.bool)
    cel = torch.stack([torch.tensor(atoms.cell).to(torch.float)
                       for atoms in mols])
    return Pnt(cel=cel, pbc=pbc, pos=pos, ent=ent)


def fill(tensors: List[Tensor], valid: Tensor):
    size = list(valid.size())
    for i in range(1, tensors[0].dim()):
        size.append(tensors[0].size()[i])
    tmp = torch.zeros(size)
    tmp[valid] = torch.cat(tensors)
    return tmp
