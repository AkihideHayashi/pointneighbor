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
    n_bch = 10
    n_pnt = 100
    n_dim = 3
    rc = 6.0

    params = [CellParameter(30.0 + i, 30.0 + i, 30.0, 1.0, 1.0, 1.0)
              for i in range(n_bch)]
    pbc = torch.rand((n_bch, n_dim)) > -1

    p = random_particle(n_bch, n_pnt, n_dim, params, pbc)
    ep = pn.exp_pcl(p)

    simple = script(pn.coo2.coo2_ful_simple)
    cel_adj = script(pn.coo2.cel_adj)(p, rc)
    get_blg = script(pn.coo2.blg)
    get_adj = script(pn.coo2.coo2_cel)
    get_vda = script(pn.vec_sod_adj)

    def using_simple():
        vda = simple(p, rc)
        return vda.adj.size()[1]

    def using_pntsft():
        blg = get_blg(cel_adj, ep)
        adj = get_adj(cel_adj, blg)
        vda = get_vda(p, adj, rc)
        return vda.adj.size()[1]

    assert using_simple() == using_pntsft(), (using_simple(), using_pntsft())
    print('simple:', timeit(using_simple))
    print('celidx:', timeit(using_pntsft))


if __name__ == "__main__":
    main()
