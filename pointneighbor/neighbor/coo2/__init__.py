# flake8: noqa
"""Make Coo-like Adjacent (2 body-problem).

Coo-like adjacent is composed of 4 arrays (n, i, j, s) and shifts.
"""
from .ful_simple import coo2_ful_simple
from .ful_pntsft import coo2_ful_pntsft
from .cel import coo2_cel, cel_num_div, cel_blg, cel_adj
from .cel import CelAdj
