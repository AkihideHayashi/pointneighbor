from typing import Optional, Union, List
import torch
from torch import nn
from ..types import AdjSftSpc, VecSod, is_coo2, PntFul
from .storage import AdjSftSpcStorage
from ..utils import cutoff_coo2, coo2_vec_sod
from ..neighbor import coo_to_lil


class Coo2Neighbor(nn.Module):
    def __init__(self, rc: float):
        super().__init__()
        self.rc = rc
        self.adj_sft_spc = AdjSftSpcStorage()

    def forward(self, adj: Optional[AdjSftSpc] = None,
                vec_sod: Optional[VecSod] = None):
        if adj is not None:
            assert vec_sod is not None
            assert is_coo2(adj)
            adj, vec_sod = cutoff_coo2(adj, vec_sod, self.rc)
            self.adj_sft_spc(adj)
        else:
            assert vec_sod is None
        return self.adj_sft_spc()

    def extra_repr(self):
        return f'{self.rc}'


class Lil2Neighbor(nn.Module):
    def __init__(self, rc: float):
        super().__init__()
        self.rc = rc
        self.adj_sft_spc = AdjSftSpcStorage()

    def forward(self, adj: Optional[AdjSftSpc] = None):
        if adj is not None:
            assert is_coo2(adj)
            adj = coo_to_lil(adj)
            self.adj_sft_spc(adj)
        return self.adj_sft_spc()

    def extra_repr(self):
        return f'{self.rc}'


class Neighbor(nn.Module):
    def __init__(self, adj: nn.Module,
                 neighbors: List[Union[Coo2Neighbor, Lil2Neighbor]]):
        super().__init__()
        for mod in neighbors:
            assert isinstance(mod, (Coo2Neighbor, Lil2Neighbor))
        coo2_list = [mod for mod in neighbors if isinstance(mod, Coo2Neighbor)]
        lil2_list = [mod for mod in neighbors if isinstance(mod, Lil2Neighbor)]
        coo2_dict = {mod.rc: mod for mod in coo2_list}
        lil2_dict = {mod.rc: mod for mod in lil2_list}
        warn = 'Duplicate cutoffs are not allowed.'
        assert len(coo2_list) == len(set(coo2_dict)), warn
        assert len(lil2_list) == len(set(lil2_dict)), warn
        for key in lil2_dict:
            if key not in coo2_dict:
                coo2_dict[key] = Coo2Neighbor(key)

        coo2_keys = sorted(coo2_dict)
        self.coo = nn.ModuleList([coo2_dict[key] for key in coo2_keys])
        self.lil = nn.ModuleList(
            [lil2_dict[key] if key in lil2_dict else torch.nn.Identity()
             for key in coo2_keys])
        self.adj = adj

    def forward(self, pf: PntFul):
        master: AdjSftSpc = self.adj(pf)
        vec_sod = coo2_vec_sod(master, pf.pos_xyz, pf.cel_mat)
        for coo, lil in zip(self.coo, self.lil):
            adj = coo(master, vec_sod)
            lil(adj)

        # for i, coo in enumerate(self.coo):
        #     coo(master, vec_sod)
        # for i, lil in enumerate(self.lil):
        #     adj_coo: AdjSftSpc = self.coo[i]()
        #     lil(adj_coo)
