import sys
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


def number(mod, pe: pn.PntExp, rc):
    adj_ = mod(pe)
    adj, _ = pn.coo2_adj_vec_sod(adj_, pe.pos_xyz, pe.cel_mat, rc)
    return adj.adj.size(1)


def main():
    if len(sys.argv) > 1:
        torch.manual_seed(int(sys.argv[1]))
    n_bch = 10
    n_pnt = 40
    n_dim = 3
    rc = 6.0
    params = [CellParameter(30.0, 30.0, 30.0, 1.0, 1.0, 1.0)
              for _ in range(n_bch)]
    pbc = torch.tensor([[True, True, True] for _ in range(n_bch)])

    ful_simple = script(pn.Coo2FulSimple(rc))
    ful_pntsft = script(pn.Coo2FulPntSft(rc))
    cel = script(pn.Coo2Cel(rc))
    bok = script(pn.Coo2BookKeeping(
        pn.Coo2FulSimple(rc + 1.0), pn.VerletCriteria(2)))

    p = random_particle(n_bch, n_pnt, n_dim, params, pbc)
    pe = pn.pnt_exp(p)
    for _ in range(100):
        noise = torch.randn_like(pe.pos_xyz) * 0.05
        p = pn.Pnt(
            pos=pe.pos_xyz + noise,
            cel=pe.cel_mat,
            pbc=pe.pbc,
            ent=pe.ent,
        )
        pe = pn.pnt_exp(p)

        n_fs = number(ful_simple, pe, rc)
        n_fp = number(ful_pntsft, pe, rc)
        n_cl = number(cel, pe, rc)
        # assert n_fs == n_fp == n_cl, (n_fs, n_fp, n_cl)
        n_bk = number(bok, pe, rc)
        assert n_fs == n_fp == n_cl == n_bk, (n_fs, n_fp, n_cl, n_bk)
        print(n_cl)


if __name__ == "__main__":
    main()
