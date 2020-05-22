import time
import torch
import pointneighbor as pn
from common import n_duo_single_lattice, random_particle, CellParameter


def timeit(f):
    s = time.time()
    f()
    e = time.time()
    return e - s


def main():
    n_bch = 20
    n_pnt = 20
    n_dim = 3
    rc = 6.0
    params = [CellParameter(11.0, 11.0, 11.0, 1.0, 1.0, 1.0)
              for _ in range(n_bch)]
    pbc = torch.tensor([[True, True, True] for _ in range(n_bch)])
    p = random_particle(n_bch, n_pnt, n_dim, params, pbc)

    def using_pn():
        pe = pn.pnt_ful(p.cel, p.pbc, p.pos, p.ent)
        vsa = pn.coo2_ful_simple(pe, rc)
        return vsa.adj.size()[1]

    def using_aev():
        return n_duo_single_lattice(p, rc)

    assert using_pn() == using_aev()
    print('torchani        :', timeit(using_aev))
    print('particleneighbor:', timeit(using_pn))


if __name__ == "__main__":
    main()
