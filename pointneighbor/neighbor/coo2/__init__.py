# flake8: noqa
"""Make Coo-like Adjacent (2 body-problem).

Coo-like adjacent is composed of 4 arrays (n, i, j, s) and shifts.


# this example is old.
Examples::
    >>> n, i, j, s = adj.adj.unbind(0)
    >>> sft_xyz = sft_cel.to(cells) @ cells

    >>> ri = pos_xyz[n, i, :]
    >>> rj = pos_xyz[n, j, :]
    >>> rs = sft_xyz[n, s, :]

    >>> vec = ri - (rj + rs)
    >>> sod = vec.pow(2).sum(dim=-1)

    >>> assert (sod <= rc * rc).all()
"""
from .ful_simple import coo2_ful_simple
from .ful_pntsft import coo2_ful_pntsft
from .cel import coo2_cel, cel_num_div, cel_blg, cel_adj
from .cel import CelAdj
