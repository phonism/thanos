import unittest
import sys
sys.path.append('../')
import thanos
import numpy as np

np.random.seed(4)

def gradient_check(f, *args, tol=1e-6, backward=False, **kwargs):
    eps = 1e-4
    numerical_grads = [np.zeros(a.shape) for a in args]
    for i in range(len(args)):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            f1 = float(f(*args, **kwargs).numpy().sum())
            args[i].realize_cached_data().flat[j] -= 2 * eps
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
    print(computed_grads, numerical_grads)
    assert error < tol
    return computed_grads

class TestOps(unittest.TestCase):
    def test_add(self):
        a = thanos.Tensor(np.asarray([[0.88282, 0.23415]]))
        b = thanos.Tensor(np.asarray([[0.18234, 0.22123]]))
        c = a + b
        sol = np.asarray([[1.06516, 0.45538]])
        np.testing.assert_allclose(c.numpy(), sol, rtol=1e-06, atol=1e-06)
        gradient_check(
                lambda A, B : A + B,
                thanos.Tensor(np.random.randn(5, 4)),
                thanos.Tensor(np.random.randn(5, 4)),
                backward=True
        )


if __name__ == '__main__':
    unittest.main()
