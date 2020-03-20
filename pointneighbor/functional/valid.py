from torch import Tensor
import torch


def val_pbc(sft_ltc: Tensor, pbc: Tensor):
    """Valid in terms of periodic boundary contition."""
    # (n_bch, n_sft, n_dim) -> (n_bch, n_sft)
    return ((sft_ltc[None, :, :] == 0) | pbc[:, None, :]).all(dim=-1)


def val_idt(sft_ltc: Tensor, n_pnt: int):
    """Valid in terms of indentity. Remove same points in same cell."""
    n_sft, _ = sft_ltc.size()
    z_sft = (sft_ltc == 0).all(dim=-1)
    size = [n_pnt, n_pnt, n_sft]
    val = sft_ltc.new_full(size, 1, dtype=torch.bool)
    val[:, :, z_sft] = ~torch.eye(n_pnt).to(sft_ltc)[:, :, None].to(torch.bool)
    return val[:, :, :]
