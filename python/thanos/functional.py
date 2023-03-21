"""functional table."""
# Global operator table.
import numpy
from .autograd import Tensor

from .backend_selection import array_api, NDArray

def triu(a: Tensor, k: int):
    return Tensor.make_const(array_api.triu(a.realize_cached_data(), k))
