import time
import torch
import pointneighbor as pn
from common import random_particle, CellParameter


def timeit(f):
    s = time.time()
    f()
    e = time.time()
    return e - s


def main():
    n_bch = 20
    n_pnt = 13
    n_dim = 3
    rc = 6.0
    params = [CellParameter(11.0, 11.0, 11.0, 1.0, 1.0, 1.0)
              for _ in range(n_bch)]
    pbc = torch.tensor([[True, True, True] for _ in range(n_bch)])
    p = random_particle(n_bch, n_pnt, n_dim, params, pbc)
    pe = pn.pnt_exp(p)
    adj = pn.coo2_ful_simple(pe, rc)
    vec_sod = pn.coo2_vec_sod(adj, pe.pos_xyz, pe.cel_mat)
    adj_lil, (vec_lil, sod_lil) = pn.lil2_adj_sft_siz_vec_sod(adj, vec_sod)
    assert torch.allclose(sod_lil, vsa_lil(adj_lil, p)[1], atol=1e-5)
    assert torch.allclose(vec_lil, vsa_lil(adj_lil, p)[0], atol=1e-5)


def vsa_lil(adj, p):
    pos = p.pos
    cel = p.cel
    j, s = adj.adj.unbind(0)
    n = pn.functional.arange_like(j, 0)
    sft_cel = adj.sft
    sft_xyz = sft_cel.to(cel) @ cel
    rj = pos[n, j]
    ri = pos[:, :, None].expand_as(rj)
    rs = sft_xyz[n, s]
    vec = pn.functional.vector(ri, rj, rs)
    sod = vec.pow(2).sum(-1)
    vec[j < 0] = 0
    sod[j < 0] = 0
    return vec, sod


if __name__ == "__main__":
    main()
