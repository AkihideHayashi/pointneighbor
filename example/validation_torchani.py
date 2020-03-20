import time
import torch
import pointneighbor as pn
from common import n_duo_multi_lattice, random_particle, CellParameter


def timeit(f):
    s = time.time()
    f()
    e = time.time()
    return e - s


def main():
    n_bch = 10
    n_pnt = 20
    n_dim = 3
    rc = 6.0
    params = [CellParameter(11.0, 11.0, 11.0, 1.0, 1.0, (i + 1) / n_bch)
              for i in range(n_bch)]
    pbc = torch.rand((n_bch, n_dim)) > 0.5
    p = random_particle(n_bch, n_pnt, n_dim, params, pbc)

    def using_pn():
        vsa = pn.coo2.coo2_ful_simple(p, rc)
        return vsa.adj.size()[1]

    def using_aev():
        return n_duo_multi_lattice(p, rc)

    assert using_pn() == using_aev()


if __name__ == "__main__":
    main()
