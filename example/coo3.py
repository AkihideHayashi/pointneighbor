import time
import torch
import pointneighbor as pn
from common import n_trio_single_lattice, random_particle, CellParameter


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
    pe = pn.pnt_exp(p)

    def using_pn():
        vsa = pn.coo2_ful_simple(pe, rc)
        jk3 = pn.coo3(vsa.adj)
        return jk3.size()[1]

    def using_aev():
        return n_trio_single_lattice(p, rc)

    assert using_pn() == using_aev()
    print('torchani        :', timeit(using_aev))
    print('particleneighbor:', timeit(using_pn))


if __name__ == "__main__":
    main()
