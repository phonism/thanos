import unittest
import sys
sys.path.append('../')
import thanos
import numpy as np

np.random.seed(19)

def gradient_check(f, *args, tol=4.2e-1, backward=False, **kwargs):
    eps = 1e-5
    numerical_grads = [np.zeros(a.shape) for a in args]
    for i in range(len(args)):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            #print(i, j)
            #print(*args)
            #print(f(*args, **kwargs))
            f1 = float(f(*args, **kwargs).numpy().sum())
            args[i].realize_cached_data().flat[j] -= 2 * eps
            #print(*args)
            #print(f(*args, **kwargs).numpy())
            f2 = float(f(*args, **kwargs).numpy().sum())
            args[i].realize_cached_data().flat[j] += eps
            numerical_grads[i].flat[j] = (f1 - f2) / (2 * eps)
    if not backward:
        out = f(*args, **kwargs)
        computed_grads = [x.numpy() for x in out.op.gradient_as_tuple(thanos.Tensor(np.ones(out.shape)), out)]
    else:
        out = f(*args, **kwargs).sum()
        out.backward()
        computed_grads = [a.grad.numpy() for a in args]
    error = sum(
        np.linalg.norm(computed_grads[i] - numerical_grads[i])
        for i in range(len(args))
    )
    #print("===============")
    #print(computed_grads, numerical_grads, "FUCK", error, "YOU", tol, error < tol)
    #print("===============")
    assert error < tol
    return computed_grads

class TestOps(unittest.TestCase):

    def test_broadcast_to(self):
        a = thanos.Tensor(np.random.rand(5, 3))
        a = thanos.ops.broadcast_to(a, (5, 3, 2, 1, 5, 4))
        np.testing.assert_allclose(a.shape, (5, 3, 2, 1, 5, 4))
        a = thanos.Tensor(np.random.rand(1, 5, 1, 4))
        a = thanos.ops.broadcast_to(a, (3, 5, 2, 4))
        np.testing.assert_allclose(a.shape, (3, 5, 2, 4))
        gradient_check(
                lambda A: A.broadcast_to((3, 4, 5, 2)),
                thanos.Tensor(np.random.randn(3, 4, 5)),
                backward=True
        )

        gradient_check(
                lambda A: A.broadcast_to((3, 4, 5)),
                thanos.Tensor(np.random.randn(3, 1, 5)),
                backward=True
        )

if __name__ == '__main__':
    unittest.main()
