# flake8: noqa
"""Make Coo-like Adjacent (2 body-problem).

Coo-like adjacent is composed of 4 arrays (n, i, j, s) and shifts.

n, i, j, s = adj.unbind(0)
sft_xyz = sft_cel.to(cells) @ cells

ri = pos_xyz[n, i, :]
rj = pos_xyz[n, j, :]
rs = sft_xyz[n, s, :]

vec = ri - (rj + rs)
sod = vec.pow(2).sum(dim=-1)

assert (sod <= rc * rc).all()
"""
from .ful_simple import coo2_ful_simple
from .ful_pntsft import coo2_ful_pntsft
from .cel_adj import cel_adj, number_division
from .cel_blg import blg
from .cel import coo2_cel
