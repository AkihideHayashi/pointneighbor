import sys
import time
import torch
from torch.jit import script
from torchani.aev import neighbor_pairs, compute_shifts
import pointneighbor as pn
from common import random_particle, CellParameter, Pnt


def timeit(f):
    s = time.time()
    f()
    e = time.time()
    return e - s


def number(mod, pe: pn.PntFul):
    adj = mod(pe)
    vec_sod = pn.coo2_vec_sod(adj, pe.pos_xyz, pe.cel_mat)
    sod = vec_sod.sod
    assert (sod <= 6 * 6).all()
    assert sod.unique(False).size(0) * 2 == sod.size(0)
    return adj.adj.size(1)


def sort(mod, pe: pn.PntFul):
    adj = mod(pe)
    vec_sod = pn.coo2_vec_sod(adj, pe.pos_xyz, pe.cel_mat)
    sod = vec_sod.sod
    return sod.sort()[0]


def using_torchani(pe: pn.PntFul):
    spc = pn.functional.get_pos_spc(pe.pos_cel, pe.pbc)
    pos_cel = pn.functional.to_unit_cell(pe.pos_cel, spc)
    assert ((pos_cel >= 0) & (pos_cel <= 1)).all()
    pos_chk = pos_cel @  pe.cel_mat
    pos = pn.functional.to_unit_cell(pe.pos_xyz, pe.spc_xyz)
    pos_cel = pos @ pe.cel_inv
    assert ((pos_cel >= 0) & (pos_cel <= 1)).all()
    assert torch.allclose(pos, pos_chk, atol=1e-2)
    print((pos - pos_chk).max())
    shifts = compute_shifts(pe.cel_mat.squeeze(0), pe.pbc.squeeze(0), 6.0)
    i, j, s = neighbor_pairs(~pe.ent, pos, pe.cel_mat.squeeze(0), shifts, 6.0,)
    return len(j) * 2


def main():
    if len(sys.argv) > 1:
        torch.manual_seed(int(sys.argv[1]))
    n_bch = 1
    n_pnt = 160
    n_dim = 3
    rc = 6.0
    params = [CellParameter(30.0, 30.0, 30.0, 1.0, 1.0, 1.0)
              for _ in range(n_bch)]
    pbc = torch.tensor([[True, True, True] for _ in range(n_bch)])

    ful_simple = script(pn.Coo2FulSimple(rc))
    ful_pntsft = script(pn.Coo2FulPntSft(rc))
    cel = script(pn.Coo2Cel(rc))
    bok = script(pn.Coo2BookKeeping(
        pn.Coo2FulSimple(rc + 2.0), pn.StrictCriteria(2.0, debug=True), rc))

    p = random_particle(n_bch, n_pnt, n_dim, params, pbc)
    pe = pn.pnt_ful(p.cel, p.pbc, p.pos, p.ent)
    for _ in range(1009):
        noise = torch.randn_like(pe.pos_xyz) * 0.1
        p = Pnt(
            pos=pe.pos_xyz + noise,
            cel=pe.cel_mat,
            pbc=pe.pbc,
            ent=pe.ent,
        )
        pe = pn.pnt_ful(p.cel, p.pbc, p.pos, p.ent)

        n_fs = number(ful_simple, pe)
        n_fp = number(ful_pntsft, pe)
        n_cl = number(cel, pe)
        n_ta = using_torchani(pe)
        s_fs = sort(ful_simple, pe)
        s_cl = sort(cel, pe)
        # assert n_fs == n_fp == n_cl, (n_fs, n_fp, n_cl)
        n_bk = number(bok, pe)
        assert n_fs == n_fp == n_cl == n_bk, (n_fs, n_fp, n_cl, n_bk, n_ta)
        assert (s_fs == s_cl).all()
        print(n_cl, n_ta, pe.spc_cel.max())


if __name__ == "__main__":
    main()
