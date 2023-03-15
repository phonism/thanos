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

        d = a + 0.1
        sol = np.asarray([[0.98282, 0.33415]])
        np.testing.assert_allclose(d.numpy(), sol, rtol=1e-06, atol=1e-06)
        gradient_check(
                lambda A : A + 0.1,
                thanos.Tensor(np.random.randn(5, 4)),
                backward=True
        )

    def test_sub(self):
        a = thanos.Tensor(np.asarray([[0.88282, 0.23415]]))
        b = thanos.Tensor(np.asarray([[0.18234, 0.22123]]))
        c = a - b
        sol = np.asarray([[0.70048, 0.01292]])
        np.testing.assert_allclose(c.numpy(), sol, rtol=1e-06, atol=1e-06)
        gradient_check(
                lambda A, B : A - B,
                thanos.Tensor(np.random.randn(5, 4)),
                thanos.Tensor(np.random.randn(5, 4)),
                backward=True
        )

        d = a - 0.1
        sol = np.asarray([[0.78282, 0.13415]])
        np.testing.assert_allclose(d.numpy(), sol, rtol=1e-06, atol=1e-06)
        gradient_check(
                lambda A : A - 0.1,
                thanos.Tensor(np.random.randn(5, 4)),
                backward=True
        )

    def test_mul(self):
        a = thanos.Tensor(np.asarray([[0.88282, 0.23415]]))
        b = thanos.Tensor(np.asarray([[0.18234, 0.22123]]))
        c = a * b
        sol = np.asarray([[0.1609733988, 0.0518010045]])
        np.testing.assert_allclose(c.numpy(), sol, rtol=1e-06, atol=1e-06)
        gradient_check(
                lambda A, B : A * B,
                thanos.Tensor(np.random.randn(5, 4)),
                thanos.Tensor(np.random.randn(5, 4)),
                backward=True
        )

        d = a * 10
        sol = np.asarray([[8.8282, 2.3415]])
        np.testing.assert_allclose(d.numpy(), sol, rtol=1e-06, atol=1e-06)
        gradient_check(
                lambda A: A * 10,
                thanos.Tensor(np.random.randn(5, 4)),
                backward=True
        )

    def test_div(self):
        a = thanos.Tensor(np.asarray([[0.88282, 0.23415]]))
        b = thanos.Tensor(np.asarray([[0.18234, 0.22123]]))
        c = a / b
        sol = np.asarray([[4.8416145661950205, 1.0584007593906792]])
        np.testing.assert_allclose(c.numpy(), sol, rtol=1e-06, atol=1e-06)
        gradient_check(
                lambda A, B : A / B,
                thanos.Tensor(np.random.randn(5, 4)),
                thanos.Tensor(5 + np.random.randn(5, 4)),
                backward=True
        )

        d = a / 10
        sol = np.asarray([[0.088282, 0.023415]])
        np.testing.assert_allclose(d.numpy(), sol, rtol=1e-06, atol=1e-06)
        gradient_check(
                lambda A: A / 10,
                thanos.Tensor(np.random.randn(5, 4)),
                backward=True
        )

    def test_pow(self):
        a = thanos.Tensor(np.asarray([[0.88282, 0.23415]]))
        c = a ** 2
        sol = np.asarray([[0.7793711524, 0.0548262225]])
        np.testing.assert_allclose(c.numpy(), sol, rtol=1e-06, atol=1e-06)
        gradient_check(
                lambda A: A ** 3,
                thanos.Tensor(np.random.randn(5, 4)),
                backward=True
        )

    def test_log(self):
        a = thanos.Tensor(np.asarray([[0.88282, 0.23415]]))
        c = a.log()
        sol = np.asarray([[-0.1246339496681425, -1.4517933433525863]])
        np.testing.assert_allclose(c.numpy(), sol, rtol=1e-06, atol=1e-06)
        gradient_check(
                lambda A: A.log(),
                thanos.Tensor(5 + np.random.randn(5, 4)),
                backward=True
        )

    def test_exp(self):
        a = thanos.Tensor(np.asarray([[0.88282, 0.23415]]))
        c = a.exp()
        sol = np.asarray([[2.4177080388261216, 1.2638340530983203]])
        np.testing.assert_allclose(c.numpy(), sol, rtol=1e-06, atol=1e-06)
        gradient_check(
                lambda A: A.exp(),
                thanos.Tensor(np.random.randn(5, 4)),
                backward=True
        )

    def test_transpose(self):
        a = thanos.Tensor(np.random.rand(3, 4, 5))
        a = thanos.ops.transpose(a, axes=None)
        np.testing.assert_allclose(a.shape, (3, 5, 4))
        a = thanos.ops.transpose(a, axes=(0, 1))
        np.testing.assert_allclose(a.shape, (5, 3, 4))
        gradient_check(
                lambda A: A.transpose(axes=(0, 1)),
                thanos.Tensor(np.random.randn(5, 4)),
                backward=True
        )
        gradient_check(
                lambda A: A.transpose(axes=None),
                thanos.Tensor(np.random.randn(5, 4)),
                backward=True
        )

    def test_reshape(self):
        a = thanos.Tensor(np.random.rand(3, 4, 5))
        a = thanos.ops.reshape(a, (3, 5, 4))
        np.testing.assert_allclose(a.shape, (3, 5, 4))
        a = thanos.ops.reshape(a, (5, 3, 2, 2))
        np.testing.assert_allclose(a.shape, (5, 3, 2, 2))
        gradient_check(
                lambda A: A.reshape((5, 3, 2, 2)),
                thanos.Tensor(np.random.randn(3, 4, 5)),
                backward=True
        )

    def test_broadcast_to(self):
        a = thanos.Tensor(np.random.rand(5, 4))
        a = thanos.ops.broadcast_to(a, (5, 4, 2, 1, 5, 4))
        np.testing.assert_allclose(a.shape, (5, 4, 2, 1, 5, 4))
        gradient_check(
                lambda A: A.broadcast_to((3, 4, 5, 2)),
                thanos.Tensor(np.random.randn(3, 4, 5)),
                backward=True
        )

        gradient_check(
                lambda A: A.broadcast_to((3, 4, 5)),
                thanos.Tensor(np.random.randn(1, 1)),
                backward=True
        )

    def test_summation(self):
        a = thanos.Tensor(np.random.rand(5, 4))
        b = thanos.ops.summation(a, (0,))
        np.testing.assert_allclose(b.shape, (4,))
        c = thanos.ops.summation(a, (1,))
        np.testing.assert_allclose(c.shape, (5,))
        d = thanos.ops.summation(a, None)
        np.testing.assert_allclose(d.shape, ())
        gradient_check(
                lambda A: A.sum((0,)),
                thanos.Tensor(np.random.randn(3, 4, 5)),
                backward=True
        )
        gradient_check(
                lambda A: A.sum(),
                thanos.Tensor(np.random.randn(3, 4, 5)),
                backward=True
        )
        gradient_check(
                lambda A: A.sum((0, 1)),
                thanos.Tensor(np.random.randn(3, 4, 5)),
                backward=True
        )

    def test_matmul(self):
        a = thanos.Tensor(np.random.rand(3, 4))
        b = thanos.Tensor(np.random.rand(4, 5))
        c = thanos.ops.matmul(a, b)
        np.testing.assert_allclose(c.shape, (3, 5))
        gradient_check(
                lambda A, B: A.matmul(B),
                a,
                b,
                backward=True
        )

    def test_matmul_tile(self):
        a = thanos.Tensor(np.random.rand(16, 24))
        b = thanos.Tensor(np.random.rand(24, 32))
        c = thanos.ops.matmul(a, b)
        np.testing.assert_allclose(c.shape, (16, 32))
        gradient_check(
                lambda A, B: A.matmul(B),
                a,
                b,
                backward=True
        )

    def test_relu(self):
        a = thanos.Tensor(np.asarray([[0.88282, -0.32]]))
        b = thanos.ops.relu(a)
        sol = thanos.Tensor(np.asarray([[0.88282, 0]]))
        np.testing.assert_allclose(b.numpy(), sol.numpy(), rtol=1e-06, atol=1e-06)
        gradient_check(
                lambda A: thanos.ops.relu(A),
                thanos.Tensor(np.random.randn(10, 9)),
                backward=True
        )

    def test_find_topo_sort(self):
        a = thanos.Tensor(np.asarray([[0.88282]]))
        b = thanos.Tensor(np.asarray([[0.18234]]))
        c = a * b + a - b / a

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
        # TODO 当前不支持二阶，可能是因为detach原因？
        #np.testing.assert_allclose(grad_x1_x1.numpy(), 2, rtol=1e-06, atol=1e-06)
        #np.testing.assert_allclose(grad_x1_x2.numpy(), 1, rtol=1e-06, atol=1e-06)

        gradient_check(
                lambda A, B, C : (A + B) * C,
                thanos.Tensor(np.random.randn(10, 9)),
                thanos.Tensor(np.random.randn(10, 9)),
                thanos.Tensor(np.random.randn(10, 9)),
                backward=True
        )


def get_tensor(*shape, entropy=1):
    np.random.seed(np.prod(shape) * len(shape) * entropy)
    return thanos.Tensor(np.random.randint(0, 100, size=shape) / 20, dtype="float32")


def linear_backward(lhs_shape, rhs_shape):
    np.random.seed(199)
    f = thanos.nn.Linear(*lhs_shape)
    f.bias.data = get_tensor(lhs_shape[-1])
    x = get_tensor(*rhs_shape)
    (f(x) ** 2).sum().backward()
    return x.grad.cached_data


class TestNN(unittest.TestCase):
    def test_module(self):
        x = thanos.Tensor(np.random.rand(128, 10))
        linear = thanos.nn.Linear(10, 50)
        x = linear(x)
        np.testing.assert_allclose(x.shape, (128, 50))

if __name__ == '__main__':
    unittest.main()
