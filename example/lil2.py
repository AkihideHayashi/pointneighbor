import time
import torch
from torch.jit import script
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
    vsa = pn.coo2_ful_simple(p, rc)
    adj = script(pn.lil2)(pn.vec_sod_adj_to_adj(vsa), p.ent.size())

    mask_coo_to_lil = pn.mask_coo_to_lil(
        pn.Adj(vsa.adj, vsa.sft), p.ent.size())

    sod_lil = pn.coo_to_lil(vsa.sod, mask_coo_to_lil, 0)
    vec_lil = pn.coo_to_lil(vsa.vec, mask_coo_to_lil, 0)
    assert (sod_lil == vsa_lil(adj, p)[1]).all()
    assert (vec_lil == vsa_lil(adj, p)[0]).all()


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
