import time
import torch
from torch.jit import script
from ase.build import bulk
import pointneighbor as pn
from common import random_particle, ase_to_particles, CellParameter


def timeit(f):
    s = time.time()
    f()
    e = time.time()
    return e - s


def random():
    n_bch = 10
    n_pnt = 13
    n_dim = 3

    params = [CellParameter(11.0, 11.0, 11.0, 1.0, 1.0, (i + 1) / n_bch)
              for i in range(n_bch)]
    pbc = torch.rand((n_bch, n_dim)) > 0.5
    p = random_particle(n_bch, n_pnt, n_dim, params, pbc)
    return p


def main():

    rc = 6.0
    atoms = [bulk('Pt') * (1, 1, 1)] + [bulk('Pt') * (4, 4, 4)] * 4
    p = ase_to_particles(atoms)

    simple = script(pn.coo2.coo2_ful_simple)
    pntsft = script(pn.coo2.coo2_ful_pntsft)
    # simple = pn.coo2.coo_duo_fullindex_simple
    # pntsft = pn.coo2.coo_duo_fullindex_pntsft

    def using_simple():
        vsa = simple(p, rc)
        return vsa.adj.size()[1]

    def using_pntsft():
        vsa = pntsft(p, rc)
        return vsa.adj.size()[1]

    assert using_simple() == using_pntsft(), (using_simple(), using_pntsft())
    print(timeit(using_simple))
    print(timeit(using_pntsft))


if __name__ == "__main__":
    main()
