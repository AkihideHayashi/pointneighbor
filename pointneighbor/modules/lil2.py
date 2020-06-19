from torch import nn
from ..type import PntFul, AdjSftSpc
from ..lil2 import lil2_adj_sft_spc


# You may want to use Lil2 module like
# Lil2(Coo2BookKeeping(Coo2Cel()))
# It is because lil2 form doe's not support cutoff reduction now.
# Maybe, reducing lil2 is as slow as coo2 -> lil2.
# Using a cutoff result with a large radius results in a slow derivative.

class Lil2(nn.Module):
    def __init__(self, coo2):
        super().__init__()
        self.coo2 = coo2

    def forward(self, pf: PntFul):
        coo2: AdjSftSpc = self.coo2(pf)
        return lil2_adj_sft_spc(coo2)
