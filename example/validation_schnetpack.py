import torch
from torch import Tensor
from schnetpack.nn import atom_distances
import pointneighbor as pn
from common import random_particle, CellParameter


def main():
    n_bch = 20
    n_pnt = 20
    n_dim = 3
    rc = 6.0
    params = [CellParameter(11.0, 11.0, 11.0, 1.0, 1.0, 1.0)
              for _ in range(n_bch)]
    pbc = torch.tensor([[True, True, True] for _ in range(n_bch)])
    p = random_particle(n_bch, n_pnt, n_dim, params, pbc, True)
    pf = pn.pnt_ful(p.cel, p.pbc, p.pos, p.ent)
    lil2 = pn.Lil2(pn.Coo2FulSimple(rc))
    adj = lil2(pf)
    vec, sod = pn.lil2_vec_sod(adj, p.pos, p.cel)
    dis: Tensor = sod.sqrt()
    j, s = pn.get_lil2_j_s(adj)
    dis_sc, vec_sc = atom_distances(p.pos, j, p.cel, s,
                                    neighbor_mask=(j >= 0).to(torch.long),
                                    return_vecs=True, normalize_vecs=False)
    vec_sc = torch.where(
        (j >= 0)[:, :, :, None].expand_as(vec_sc),
        vec_sc, torch.zeros_like(vec_sc)
    )
    assert torch.allclose(dis, dis_sc, atol=1e-5)
    assert torch.allclose(vec, vec_sc, atol=1e-5)


if __name__ == "__main__":
    main()
