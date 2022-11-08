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
    assert error < tol
    return computed_grads


class MyTest(unittest.TestCase):
    def test_add(self):
        a = thanos.Tensor(np.asarray([[0.88282, 0.23415]]))
        b = thanos.Tensor(np.asarray([[0.18234, 0.22123]]))
        c = a + b
        sol = np.asarray([[1.06516, 0.45538]])
        np.testing.assert_allclose(c.numpy(), sol, rtol=1e-06, atol=1e-06)
        d = a + 0.1
        sol = np.asarray([[0.98282, 0.33415]])
        np.testing.assert_allclose(d.numpy(), sol, rtol=1e-06, atol=1e-06)

    def test_sub(self):
        a = thanos.Tensor(np.asarray([[0.88282, 0.23415]]))
        b = thanos.Tensor(np.asarray([[0.18234, 0.22123]]))
        c = a - b
        sol = np.asarray([[0.70048, 0.01292]])
        np.testing.assert_allclose(c.numpy(), sol, rtol=1e-06, atol=1e-06)
        d = a - 0.1
        sol = np.asarray([[0.78282, 0.13415]])
        np.testing.assert_allclose(d.numpy(), sol, rtol=1e-06, atol=1e-06)

    def test_mul(self):
        a = thanos.Tensor(np.asarray([[0.88282, 0.23415]]))
        b = thanos.Tensor(np.asarray([[0.18234, 0.22123]]))
        c = a * b
        sol = np.asarray([[0.1609733988, 0.0518010045]])
        np.testing.assert_allclose(c.numpy(), sol, rtol=1e-06, atol=1e-06)
        d = a * 10
        sol = np.asarray([[8.8282, 2.3415]])
        np.testing.assert_allclose(d.numpy(), sol, rtol=1e-06, atol=1e-06)

    def test_div(self):
        a = thanos.Tensor(np.asarray([[0.88282, 0.23415]]))
        b = thanos.Tensor(np.asarray([[0.18234, 0.22123]]))
        c = a / b
        sol = np.asarray([[4.8416145661950205, 1.0584007593906792]])
        np.testing.assert_allclose(c.numpy(), sol, rtol=1e-06, atol=1e-06)
        d = a / 10
        sol = np.asarray([[0.088282, 0.023415]])
        np.testing.assert_allclose(d.numpy(), sol, rtol=1e-06, atol=1e-06)

    def test_pow(self):
        a = thanos.Tensor(np.asarray([[0.88282, 0.23415]]))
        c = a ** 2
        sol = np.asarray([[0.7793711524, 0.0548262225]])
        np.testing.assert_allclose(c.numpy(), sol, rtol=1e-06, atol=1e-06)

    def test_transpose(self):
        a = thanos.Tensor(np.random.rand(3, 4, 5))
        a = thanos.ops.transpose(a, axes=None)
        np.testing.assert_allclose(a.shape, (3, 5, 4))
        a = thanos.ops.transpose(a, axes=(0, 1))
        np.testing.assert_allclose(a.shape, (5, 3, 4))

    def test_reshape(self):
        a = thanos.Tensor(np.random.rand(3, 4, 5))
        a = thanos.ops.reshape(a, (3, 5, 4))
        np.testing.assert_allclose(a.shape, (3, 5, 4))
        a = thanos.ops.reshape(a, (5, 3, 2, 2))
        np.testing.assert_allclose(a.shape, (5, 3, 2, 2))

    def test_find_topo_sort(self):
        a = thanos.Tensor(np.asarray([[0.88282]]))
        b = thanos.Tensor(np.asarray([[0.18234]]))
        c = a * b + a - b / a
        #print(thanos.autograd.find_topo_sort(c, 1))
        c.backward()

    def test_compute_gradient_of_variables(self):
        x1 = thanos.Tensor([6])
        x2 = thanos.Tensor([0])
        y = x1 * x1 + x1 * x2
        y.backward()
        grad_x1 = x1.grad
        grad_x2 = x2.grad

        grad_x1.backward()
        grad_x1_x1 = x1.grad
        grad_x1_x2 = x2.grad

        np.testing.assert_allclose(y.numpy(), x1.numpy() * x1.numpy() + x1.numpy() * x2.numpy(), rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(grad_x1.numpy(), 2 * x1.numpy() + x2.numpy(), rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(grad_x2.numpy(), x1.numpy(), rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(grad_x1_x1.numpy(), 2, rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(grad_x1_x2.numpy(), 1, rtol=1e-06, atol=1e-06)


        #gradient_check(
                #lambda A, B, C : (A + B) * (B / C),
                #thanos.Tensor(np.random.randn(10, 9)),
                #thanos.Tensor(np.random.randn(10, 9)),
                #thanos.Tensor(np.random.randn(10, 9)), backward=True)

if __name__ == '__main__':
    unittest.main()
