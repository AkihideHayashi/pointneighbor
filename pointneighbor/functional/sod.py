from torch import Tensor


def vector(ri: Tensor, rj: Tensor, rs: Tensor):
    """rij = ri - (rj + rs)"""
    return ri - (rj + rs)


def square_of_distance(vec: Tensor, dim: int = -1):
    return (vec * vec).sum(dim=dim)


def cutoff_valid(sod: Tensor, rc: float) -> Tensor:
    return sod < rc * rc
