#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace thanos {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
    AlignedArray(const size_t size) {
        int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
        if (ret != 0) throw std::bad_alloc();
        this->size = size;
    }
    ~AlignedArray() { 
        free(ptr); 
    }
    size_t ptr_as_int() {
        return (size_t)ptr; 
    }
    scalar_t* ptr;
    size_t size;
};


void Fill(AlignedArray* out, scalar_t val) {
    /**
     * Fill the values of an aligned array with val
     */
    for (int i = 0; i < out->size; i++) {
        out->ptr[i] = val;
    }
}


void Compact(const AlignedArray& a, AlignedArray* out, 
        std::vector<uint32_t> shape, std::vector<uint32_t> strides, size_t offset) {
    /**
     * Compact an array in memory
     *
     * Args:
     *   a: non-compact representation of the array, given as input
     *   out: compact version of the array to be written
     *   shape: shapes of each dimension for a and out
     *   strides: strides of the *a* array (not out, which has compact strides)
     *   offset: offset of the *a* array (not out, which has zero offset, being compact)
     *
     * Returns:
     *  void (you need to modify out directly, rather than returning anything; this is true for all the
     *  function will implement here, so we won't repeat this note.)
     */
    uint32_t dim = shape.size();
    uint32_t total_cnt = 1;
    for (uint32_t sp : shape) {
        total_cnt *= sp;
    }

    std::vector<uint32_t> idx(dim, 0);
    for (int cnt = 0; cnt < total_cnt; ++cnt) {
        uint32_t cur_idx = offset;
        for (int i = 0; i < dim; ++i) {
            cur_idx += strides[i] * idx[i];
        } 
        out->ptr[cnt] = a.ptr[cur_idx];
        idx[dim - 1]++;

        for (int i = dim - 1; i >= 0; i--) {
            if (idx[i] >= shape[i]) {
                idx[i] = 0;
                if (i > 0) {
                    idx[i - 1]++;
                }
            }
        }
    }
}

void EwiseSetitem(const AlignedArray& a, AlignedArray* out, 
        std::vector<uint32_t> shape, std::vector<uint32_t> strides, size_t offset) {
    /**
     * Set items in a (non-compact) array
     *
     * Args:
     *   a: _compact_ array whose items will be written to out
     *   out: non-compact array whose items are to be written
     *   shape: shapes of each dimension for a and out
     *   strides: strides of the *out* array (not a, which has compact strides)
     *   offset: offset of the *out* array (not a, which has zero offset, being compact)
     */
    uint32_t dim = shape.size();
    uint32_t total_cnt = 1;
    for (uint32_t sp : shape) {
        total_cnt *= sp;
    }

    std::vector<uint32_t> idx(dim, 0);
    for (int cnt = 0; cnt < total_cnt; ++cnt) {
        uint32_t cur_idx = offset;
        for (int i = 0; i < dim; ++i) {
            cur_idx += strides[i] * idx[i];
        }
        out->ptr[cur_idx] = a.ptr[cnt];

        idx[dim - 1]++;

        for (int i = dim - 1; i >= 0; i--) {
            if (idx[i] >= shape[i]) {
                idx[i] = 0;
                if (i > 0) {
                    idx[i - 1]++;
                }
            }
        }
    }
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, 
        std::vector<uint32_t> shape, std::vector<uint32_t> strides, size_t offset) {
    /**
     * Set items is a (non-compact) array
     *
     * Args:
     *   size: number of elements to write in out array (note that this will note be the same as
     *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
     *         product of items in shape, but convenient to just pass it here.
     *   val: scalar value to write to
     *   out: non-compact array whose items are to be written
     *   shape: shapes of each dimension of out
     *   strides: strides of the out array
     *   offset: offset of the out array
     */
    uint32_t dim = shape.size();
    uint32_t total_cnt = 1;
    for (uint32_t sp : shape) {
        total_cnt *= sp;
    }

    std::vector<uint32_t> idx(dim, 0);
    for (int cnt = 0; cnt < total_cnt; ++cnt) {
        uint32_t cur_idx = offset;
        for (int i = 0; i < dim; ++i) {
            cur_idx += strides[i] * idx[i];
        }
        out->ptr[cur_idx] = val;

        idx[dim - 1]++;

        for (int i = dim - 1; i >= 0; i--) {
            if (idx[i] >= shape[i]) {
                idx[i] = 0;
                if (i > 0) {
                    idx[i - 1]++;
                }
            }
        }
    }
}

void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
    /**
     * Set entries in out to be the sum of correspondings entires in a and b.
     */
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = a.ptr[i] + b.ptr[i];
    }
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
    /**
     * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
     */
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = a.ptr[i] + val;
    }
}


/**
 * In the code the follows, use the above template to create analogous element-wise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

void EwiseMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
    /**
     * Set entries in out to be the multiply of correspondings entires in a and b.
     */
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = a.ptr[i] * b.ptr[i];
    }
}

void ScalarMul(const AlignedArray& a, scalar_t val, AlignedArray* out) {
    /**
     * Set entries in out to be the multiply of corresponding entry in a plus the scalar val.
     */
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = a.ptr[i] * val;
    }
}

void EwiseDiv(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
    /**
     * Set entries in out to be the multiply of correspondings entires in a and b.
     */
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = a.ptr[i] / b.ptr[i];
    }
}

void ScalarDiv(const AlignedArray& a, scalar_t val, AlignedArray* out) {
    /**
     * Set entries in out to be the multiply of corresponding entry in a plus the scalar val.
     */
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = a.ptr[i] / val;
    }
}

void ScalarPower(const AlignedArray& a, scalar_t val, AlignedArray* out) {
    /**
     * Set entries in out to be the multiply of corresponding entry in a plus the scalar val.
     */
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = std::pow(a.ptr[i], val);
    }
}

void EwiseMaximum(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
    /**
     * Set entries in out to be the multiply of correspondings entires in a and b.
     */
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = std::max(a.ptr[i], b.ptr[i]);
    }
}

void ScalarMaximum(const AlignedArray& a, scalar_t val, AlignedArray* out) {
    /**
     * Set entries in out to be the multiply of corresponding entry in a plus the scalar val.
     */
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = std::max(a.ptr[i], val);
    }
}

void EwiseEq(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
    /**
     * Set entries in out to be the multiply of correspondings entires in a and b.
     */
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = (scalar_t)(a.ptr[i] == b.ptr[i]);
    }
}

void ScalarEq(const AlignedArray& a, scalar_t val, AlignedArray* out) {
    /**
     * Set entries in out to be the multiply of corresponding entry in a plus the scalar val.
     */
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = (scalar_t)(a.ptr[i] == val);
    }
}

void EwiseGe(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
    /**
     * Set entries in out to be the multiply of correspondings entires in a and b.
     */
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = (scalar_t)(a.ptr[i] >= b.ptr[i]);
    }
}

void ScalarGe(const AlignedArray& a, scalar_t val, AlignedArray* out) {
    /**
     * Set entries in out to be the multiply of corresponding entry in a plus the scalar val.
     */
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = (scalar_t)(a.ptr[i] >= val);
    }
}

void EwiseSin(const AlignedArray& a, AlignedArray* out) {
    /**
     * Set entries in out to be the multiply of correspondings entires in a and b.
     */
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = std::sin(a.ptr[i]);
    }
}

void EwiseCos(const AlignedArray& a, AlignedArray* out) {
    /**
     * Set entries in out to be the multiply of correspondings entires in a and b.
     */
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = std::cos(a.ptr[i]);
    }
}

void EwiseLog(const AlignedArray& a, AlignedArray* out) {
    /**
     * Set entries in out to be the multiply of correspondings entires in a and b.
     */
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = std::log(a.ptr[i]);
    }
}

void EwiseExp(const AlignedArray& a, AlignedArray* out) {
    /**
     * Set entries in out to be the multiply of correspondings entires in a and b.
     */
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = std::exp(a.ptr[i]);
    }
}

void EwiseTanh(const AlignedArray& a, AlignedArray* out) {
    /**
     * Set entries in out to be the multiply of correspondings entires in a and b.
     */
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = std::tanh(a.ptr[i]);
    }
}

void EwiseSqrt(const AlignedArray& a, AlignedArray* out) {
    /**
     * Set entries in out to be the multiply of correspondings entires in a and b.
     */
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = std::sqrt(a.ptr[i]);
    }
}

/// END YOUR SOLUTION

void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, 
        uint32_t pre_a, uint32_t pre_b, uint32_t m, uint32_t n, uint32_t p) {
    /**
     * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
     * you can use the "naive" three-loop algorithm.
     *
     * Args:
     *   a: compact 2D array of size m x n
     *   b: compact 2D array of size n x p
     *   out: compact 2D array of size m x p to write the output to
     *   m: rows of a / out
     *   n: columns of a / rows of b
     *   p: columns of b / out
     */
    for (size_t l = 0; l < std::max(pre_a, pre_b); ++l) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < p; ++j) {
                out->ptr[l * m * p + i * p + j] = 0;
                for (size_t k = 0; k < n; ++k) {
                    size_t aa = (pre_a == 1 ? 0 : l);
                    size_t bb = (pre_b == 1 ? 0 : l);
                    out->ptr[l * m * p + i * p + j] += a.ptr[aa * m * n + i * n + k] * b.ptr[bb * n * p + k * p + j];
                }
            }
        }
    }
}

inline void AlignedDot(
        const scalar_t* __restrict__ a, const scalar_t* __restrict__ b,
        scalar_t* __restrict__ out) {
    /**
     * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
     * the result to the existing out, which you should not set to zero beforehand).  We are including
     * the compiler flags here that enable the compile to properly use vector operators to implement
     * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
     * out don't have any overlapping memory (which is necessary in order for vector operations to be
     * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
     * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
     * compiler that the input array will be aligned to the appropriate blocks in memory, which also
     * helps the compiler vectorize the code.
     *
     * Args:
     *   a: compact 2D array of size TILE x TILE
     *   b: compact 2D array of size TILE x TILE
     *   out: compact 2D array of size TILE x TILE to write to
     */
    a = (const scalar_t*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
    b = (const scalar_t*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
    out = (scalar_t*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

    for (size_t i = 0; i < TILE; ++i) {
        for (size_t j = 0; j < TILE; ++j) {
            for (size_t k = 0; k < TILE; ++k) {
                out[i * TILE + j] += a[i * TILE + k] * b[k * TILE + j];
            }
        }
    }
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, 
        uint32_t pre_a, uint32_t pre_b, uint32_t m, uint32_t n, uint32_t p) {
    /**
     * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
     * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
     *   a[m/TILE][n/TILE][TILE][TILE]
     * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
     * function should call `AlignedDot()` implemented above).
     *
     * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
     * assume that this division happens without any remainder.
     *
     * Args:
     *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
     *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
     *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
     *   m: rows of a / out
     *   n: columns of a / rows of b
     *   p: columns of b / out
     *
     */
    for (size_t lb = 0; lb < std::max(pre_a, pre_b); ++lb) {
        for (size_t i = 0; i < m / TILE; ++i) {
            scalar_t A[TILE * n];
            for (size_t ii = 0; ii < n / TILE; ++ii) {
                for (size_t jj = 0; jj < TILE * TILE; ++jj) {
                    size_t aa = (pre_a == 1 ? 0 : lb);
                    A[ii * TILE * TILE + jj] = a.ptr[aa * m * n + i * n * TILE + ii * TILE * TILE + jj];
                }
            }
            for (size_t j = 0; j < p / TILE; ++j) {
                scalar_t B[TILE * n];
                for (size_t ii = 0; ii < n / TILE; ++ii) {
                    for (size_t jj = 0; jj < TILE * TILE; ++jj) {
                        size_t bb = (pre_b == 1 ? 0 : lb);
                        B[ii * TILE * TILE + jj] = b.ptr[bb * n * p + j * TILE * TILE + ii * TILE * p + jj];
                    }
                }
                for (int k = 0; k < TILE; ++k) {
                    for (int l = 0; l < TILE; ++l) {
                        out->ptr[lb * m * p + i * p * TILE + j * TILE * TILE + k * TILE + l] = 0;
                    }
                }
                for (int k = 0; k < n / TILE; ++k) {
                    scalar_t C[TILE * TILE];
                    for (size_t ii = 0; ii < TILE * TILE; ++ii) {
                        C[ii] = 0;
                    }
                    AlignedDot(&A[k * TILE * TILE], &B[k * TILE * TILE], C);
                    for (size_t ii = 0; ii < TILE * TILE; ++ii) {
                        out->ptr[lb * m * p + i * p * TILE + j * TILE * TILE + ii] += C[ii];
                    }
                }
            }
        }
    }
    for (size_t i = 0; i < m / TILE; ++i) {
        scalar_t A[TILE * n];
        for (size_t ii = 0; ii < n / TILE; ++ii) {
            for (size_t jj = 0; jj < TILE * TILE; ++jj) {
                A[ii * TILE * TILE + jj] = a.ptr[i * n * TILE + ii * TILE * TILE + jj];
            }
        }
        for (size_t j = 0; j < p / TILE; ++j) {
            scalar_t B[TILE * n];
            for (size_t ii = 0; ii < n / TILE; ++ii) {
                for (size_t jj = 0; jj < TILE * TILE; ++jj) {
                    B[ii * TILE * TILE + jj] = b.ptr[j * TILE * TILE + ii * TILE * p + jj];
                }
            }
            for (int k = 0; k < TILE; ++k) {
                for (int l = 0; l < TILE; ++l) {
                    out->ptr[i * p * TILE + j * TILE * TILE + k * TILE + l] = 0;
                }
            }
            for (int k = 0; k < n / TILE; ++k) {
                scalar_t C[TILE * TILE];
                for (size_t ii = 0; ii < TILE * TILE; ++ii) {
                    C[ii] = 0;
                }
                AlignedDot(&A[k * TILE * TILE], &B[k * TILE * TILE], C);
                for (size_t ii = 0; ii < TILE * TILE; ++ii) {
                    out->ptr[i * p * TILE + j * TILE * TILE + ii] += C[ii];
                }
            }
        }
    }
}


void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
    /**
     * Reduce by taking maximum over `reduce_size` contiguous blocks.
     *
     * Args:
     *   a: compact array of size a.size = out.size * reduce_size to reduce over
     *   out: compact array to write into
     *   reduce_size: size of the dimension to reduce over
     */
    scalar_t tmp_max = 0;
    for (int i = 0; i < out->size; ++i) {
        tmp_max = a.ptr[i * reduce_size];
        for (int j = 0; j < reduce_size; ++j) {
            tmp_max = std::fmax(tmp_max, a.ptr[i * reduce_size + j]);
        }
        out->ptr[i] = tmp_max;
    }
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
    /**
     * Reduce by taking sum over `reduce_size` contiguous blocks.
     *
     * Args:
     *   a: compact array of size a.size = out.size * reduce_size to reduce over
     *   out: compact array to write into
     *   reduce_size: size of the dimension to reduce over
     */
    scalar_t tmp_sum = 0;
    for (int i = 0; i < out->size; ++i) {
        tmp_sum = 0;
        for (int j = 0; j < reduce_size; ++j) {
            tmp_sum += a.ptr[i * reduce_size + j];
        }
        out->ptr[i] = tmp_sum;
    }
}

/**
 * return diag
 *
 * Args:
 *   a: compact array of size a.size = out.size * reduce_size to reduce over
 *   out: compact array to write into
 */
void Diag(const AlignedArray& a, AlignedArray* out, std::vector<uint32_t>& shape, std::vector<uint32_t>& strides) {
    uint32_t dim = shape.size();
    if (dim > 1) {
        uint32_t total_cnt = 1;
        for (size_t i = 0; i < shape.size(); ++i) {
            total_cnt *= shape[i];
        }

        size_t idx = 0;
        for (size_t cnt = 0; cnt < total_cnt; ++cnt) {
            if (cnt % shape[dim - 1] == cnt / shape[dim - 1]) {
                out->ptr[idx] = a.ptr[cnt];
                idx++;
            }
        }
    } else {
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[0]; ++j) {
                if (i == j) {
                    out->ptr[i * shape[0] + j] = a.ptr[i];
                } else {
                    out->ptr[i * shape[0] + j] = 0;
                }
            }
        }
    }
}

/**
 * return triu
 *
 * Args:
 *   a: compact array of size a.size = out.size * reduce_size to reduce over
 *   out: compact array to write into
 */
void Triu(const AlignedArray& a, AlignedArray* out, std::vector<uint32_t>& shape, std::vector<uint32_t>& strides, int k) {
    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            if (i - j > -k) {
                out->ptr[i * strides[0] + j] = 0;
            } else {
                out->ptr[i * strides[0] + j] = a.ptr[i * strides[0] + j];
            }

        }
    }
}

}  // namespace cpu
}  // namespace thanos

PYBIND11_MODULE(ndarray_backend_cpu, m) {
    namespace py = pybind11;
    using namespace thanos;
    using namespace cpu;

    m.attr("__device_name__") = "cpu";
    m.attr("__tile_size__") = TILE;

    py::class_<AlignedArray>(m, "Array")
        .def(py::init<size_t>(), py::return_value_policy::take_ownership)
        .def("ptr", &AlignedArray::ptr_as_int)
        .def_readonly("size", &AlignedArray::size);

    // return numpy array (with copying for simplicity, otherwise garbage
    // collection is a pain)
    m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                std::vector<size_t> strides, size_t offset) {
            std::vector<size_t> numpy_strides = strides;
        std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                [](size_t& c) { return c * ELEM_SIZE; });
        return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
    });

    // convert from numpy (with copying)
    m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
        std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
    });

    m.def("fill", Fill);
    m.def("compact", Compact);
    m.def("ewise_setitem", EwiseSetitem);
    m.def("scalar_setitem", ScalarSetitem);
    m.def("ewise_add", EwiseAdd);
    m.def("scalar_add", ScalarAdd);
    m.def("ewise_mul", EwiseMul);
    m.def("scalar_mul", ScalarMul);
    m.def("ewise_div", EwiseDiv);
    m.def("scalar_div", ScalarDiv);
    m.def("scalar_power", ScalarPower);
    m.def("ewise_maximum", EwiseMaximum);
    m.def("scalar_maximum", ScalarMaximum);
    m.def("ewise_eq", EwiseEq);
    m.def("scalar_eq", ScalarEq);
    m.def("ewise_ge", EwiseGe);
    m.def("scalar_ge", ScalarGe);
    m.def("ewise_sin", EwiseSin);
    m.def("ewise_cos", EwiseCos);
    m.def("ewise_log", EwiseLog);
    m.def("ewise_exp", EwiseExp);
    m.def("ewise_tanh", EwiseTanh);
    m.def("ewise_sqrt", EwiseSqrt);
    m.def("matmul", Matmul);
    m.def("matmul_tiled", MatmulTiled);
    m.def("reduce_max", ReduceMax);
    m.def("reduce_sum", ReduceSum);
    m.def("diag", Diag);
    m.def("triu", Triu);
}
