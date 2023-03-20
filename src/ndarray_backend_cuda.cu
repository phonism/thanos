#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256
#define TILE 4
typedef double scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

__device__ scalar_t ADD(scalar_t a, scalar_t b) {return a + b;}
__device__ scalar_t DIV(scalar_t a, scalar_t b) {return a / b;}
__device__ scalar_t MUL(scalar_t a, scalar_t b) {return a * b;}
__device__ scalar_t EQ(scalar_t a, scalar_t b) {return a == b;}
__device__ scalar_t GE(scalar_t a, scalar_t b) {return a >= b;}
__device__ scalar_t POWER(scalar_t a, scalar_t b) {return std::pow(a, b);}
__device__ scalar_t LOG(scalar_t a) {return std::log(a);}
__device__ scalar_t EXP(scalar_t a) {return std::exp(a);}
__device__ scalar_t TANH(scalar_t a) {return std::tanh(a);}
__device__ scalar_t MAX(scalar_t a, scalar_t b) {return a > b ? a : b;}

struct CudaArray {
    CudaArray(const size_t size) {
        cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
        this->size = size;
    }
    ~CudaArray() { 
        cudaFree(ptr); 
    }
    size_t ptr_as_int() { 
        return (size_t)ptr; 
    }
  
    scalar_t* ptr;
    size_t size;
};

struct CudaDims {
    dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
    /**
     * Utility function to get cuda dimensions for 1D call
     */
    CudaDims dim;
    size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
    dim.block = dim3(BASE_THREAD_NUM, 1, 1);
    dim.grid = dim3(num_blocks, 1, 1);
    return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
    uint32_t size;
    uint32_t data[MAX_VEC_SIZE];
};


CudaVec VecToCuda(const std::vector<uint32_t>& x) {
    CudaVec shape;
    if (x.size() > MAX_VEC_SIZE) {
        throw std::runtime_error("Exceeded CUDA supported max dimesions");
    }
    shape.size = x.size();
    for (size_t i = 0; i < x.size(); i++) {
        shape.data[i] = x[i];
    }
    return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
    CudaDims dim = CudaOneDim(out->size);
    FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides

__device__ size_t getIndex(CudaVec shape, size_t gid, CudaVec strides) {
    CudaVec out_cumprod = {1, {1}};
    size_t size = shape.size;
    for (size_t i = size-1; i > 0; i--) {
        out_cumprod.data[out_cumprod.size] = shape.data[i]*out_cumprod.data[out_cumprod.size-1];
        out_cumprod.size ++;
    } 

    size_t prod = out_cumprod.data[out_cumprod.size-1];
    size_t pre_seq_gid = gid / prod;
    size_t out_gid = pre_seq_gid * strides.data[0];
    CudaVec out_gid_seq = {1, {1}};

    out_cumprod.size --;

    while (out_cumprod.size) {
        // printf("getIndex: out_cumprod: %d ", out_cumprod.size);
        out_gid_seq.data[out_gid_seq.size] = gid / out_cumprod.data[out_cumprod.size-1];
        out_gid_seq.size ++;
    
        size_t seq_gid = out_gid_seq.data[out_gid_seq.size-1] - pre_seq_gid * shape.data[size-out_cumprod.size];
        pre_seq_gid = out_gid_seq.data[out_gid_seq.size-1];
        prod = out_cumprod.data[out_cumprod.size-1];
        out_gid += strides.data[size-out_cumprod.size] * seq_gid;
    
        out_cumprod.size --;
    }
    
    return out_gid;
}


__global__ void CompactKernel(
        const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
        CudaVec strides, size_t offset) {
    /**
     * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
     * non-compact input a, to the corresponding item (at location gid) in the compact array out.
     * 
     * Args:
     *   a: CUDA pointer to a array
     *   out: CUDA point to out array
     *   size: size of out array
     *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
     *   strides: vector of strides of out array
     *   offset: offset of out array
     */
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < size) {
        out[gid] = a[offset+getIndex(shape, gid, strides)];
    }
}

void Compact(
        const CudaArray& a, CudaArray* out, std::vector<uint32_t> shape,
        std::vector<uint32_t> strides, size_t offset) {
    /**
     * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
     * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
     * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
     * the functions after this, however, you'll need to define these kernels as you see fit to 
     * execute the underlying function.
     * 
     * Args:
     *   a: non-compact represntation of the array, given as input
     *   out: compact version of the array to be written
     *   shape: shapes of each dimension for a and out
     *   strides: strides of the *a* array (not out, which has compact strides)
     *   offset: offset of the *a* array (not out, which has zero offset, being compact) 
     */

    // Nothing needs to be added here
    CudaDims dim = CudaOneDim(out->size);
    CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
            VecToCuda(strides), offset);
}

__global__ void EwiseSetitemKernel(
        size_t size, const scalar_t* a, scalar_t* out, CudaVec shape,
        CudaVec strides, size_t offset) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        out[offset+getIndex(shape, gid, strides)] = a[gid];
    }
}

void EwiseSetitem(
        const CudaArray& a, CudaArray* out, std::vector<uint32_t> shape,
        std::vector<uint32_t> strides, size_t offset) {
    /**
     * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
     * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
     * 
     * Args:
     *   a: _compact_ array whose items will be written to out
     *   out: non-compact array whose items are to be written
     *   shape: shapes of each dimension for a and out
     *   strides: strides of the *out* array (not a, which has compact strides)
     *   offset: offset of the *out* array (not a, which has zero offset, being compact)
     */
    CudaDims dim = CudaOneDim(a.size);
    EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.size, a.ptr, out->ptr, VecToCuda(shape),
            VecToCuda(strides), offset);
}

__global__ void ScalarSetitemKernel(
        size_t size, scalar_t val, scalar_t* out, CudaVec shape,
        CudaVec strides, size_t offset) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        out[offset+getIndex(shape, gid, strides)] = val;
    }
}

void ScalarSetitem(
        size_t size, scalar_t val, CudaArray* out, std::vector<uint32_t> shape,
        std::vector<uint32_t> strides, size_t offset) {
    /**
     * Set items is a (non-compact) array
     * 
     * Args:
     *   size: number of elements to write in out array (note that this will note be the same as
     *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
     *         product of items in shape, but covenient to just pass it here.
     *   val: scalar value to write to
     *   out: non-compact array whose items are to be written
     *   shape: shapes of each dimension of out
     *   strides: strides of the out array
     *   offset: offset of the out array
     */
    CudaDims dim = CudaOneDim(size);
    ScalarSetitemKernel<<<dim.grid, dim.block>>>(size, val, out->ptr, VecToCuda(shape),
            VecToCuda(strides), offset);
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__device__ void EwiseOperatorKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size, scalar_t(* op)(scalar_t, scalar_t)) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        out[gid] = op(a[gid], b[gid]);
    }
}

__device__ void EwiseOperatorKernel(const scalar_t* a, scalar_t* out, size_t size, scalar_t(* op)(scalar_t)) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        out[gid] = op(a[gid]);
    }
}

__device__ void ScalarOperatorKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size, scalar_t(* op)(scalar_t, scalar_t)) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        out[gid] = op(a[gid], val);
    }
}

__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
    EwiseOperatorKernel(a, b, out, size, ADD);
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
    /**
     * Add together two CUDA array
     */
    CudaDims dim = CudaOneDim(out->size);
    EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
    ScalarOperatorKernel(a, val, out, size, ADD);
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
    /**
     * Add together a CUDA array and a scalar value.
     */
    CudaDims dim = CudaOneDim(out->size);
    ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
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

/// BEGIN YOUR SOLUTION
__global__ void EwiseMulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
    EwiseOperatorKernel(a, b, out, size, MUL);
}

void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
    CudaDims dim = CudaOneDim(out->size);
    EwiseMulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMulKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
    ScalarOperatorKernel(a, val, out, size, MUL);
}

void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
    CudaDims dim = CudaOneDim(out->size);
    ScalarMulKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseDivKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
    EwiseOperatorKernel(a, b, out, size, DIV);
}

void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out) {
    CudaDims dim = CudaOneDim(out->size);
    EwiseDivKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarDivKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
    ScalarOperatorKernel(a, val, out, size, DIV);
}

void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {
    CudaDims dim = CudaOneDim(out->size);
    ScalarDivKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseMaxKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
    EwiseOperatorKernel(a, b, out, size, MAX);
}

void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
    CudaDims dim = CudaOneDim(out->size);
    EwiseMaxKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMaxKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
    ScalarOperatorKernel(a, val, out, size, MAX);
}

void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {
    CudaDims dim = CudaOneDim(out->size);
    ScalarMaxKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void ScalarPowerKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
    ScalarOperatorKernel(a, val, out, size, POWER);
}

void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
    CudaDims dim = CudaOneDim(out->size);
    ScalarPowerKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseEqKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
    EwiseOperatorKernel(a, b, out, size, EQ);
}

void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
    CudaDims dim = CudaOneDim(out->size);
    EwiseEqKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarEqKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
    ScalarOperatorKernel(a, val, out, size, EQ);
}

void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out) {
    CudaDims dim = CudaOneDim(out->size);
    ScalarEqKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseGeKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
    EwiseOperatorKernel(a, b, out, size, GE);
}

void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
    CudaDims dim = CudaOneDim(out->size);
    EwiseGeKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarGeKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
    ScalarOperatorKernel(a, val, out, size, GE);
}

void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out) {
    CudaDims dim = CudaOneDim(out->size);
    ScalarGeKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseLogKernel(const scalar_t* a, scalar_t* out, size_t size) {
    EwiseOperatorKernel(a, out, size, LOG);
}

void EwiseLog(const CudaArray& a, CudaArray* out) {
    CudaDims dim = CudaOneDim(out->size);
    EwiseLogKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void EwiseExpKernel(const scalar_t* a, scalar_t* out, size_t size) {
    EwiseOperatorKernel(a, out, size, EXP);
}

void EwiseExp(const CudaArray& a, CudaArray* out) {
    CudaDims dim = CudaOneDim(out->size);
    EwiseExpKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void EwiseTanhKernel(const scalar_t* a, scalar_t* out, size_t size) {
    EwiseOperatorKernel(a, out, size, TANH);
}

void EwiseTanh(const CudaArray& a, CudaArray* out) {
    CudaDims dim = CudaOneDim(out->size);
    EwiseTanhKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void MatmulTileKernel(
        scalar_t* a, scalar_t* b, scalar_t* out, 
        uint32_t pre_a, uint32_t pre_b,
        uint32_t M, uint32_t N, uint32_t P) {
    // Shared memory used to store Asub and Bsub respectively 
    __shared__ scalar_t As[TILE][TILE];
    __shared__ scalar_t Bs[TILE][TILE];

    // Block row and column
    size_t bidc = blockIdx.y * TILE;
    size_t bidr = blockIdx.x * TILE;
    size_t pre = blockIdx.z;
 
    // Accumulate results into Outvalue 
    scalar_t Outvalue = 0;

    // Thread row and column within Out_sub  
    uint32_t tidc = threadIdx.y;
    uint32_t tidr = threadIdx.x;

    // Look over all the sub-matrices of A and B that are required to compute the sub-matrice of Out
    // Multiply each pair of sub-matrices together and accumulate the results 
    for (size_t i = 0; i < N; i += TILE) {
        // Load Asub and Bsub from device memory to shared memory  
        // Each thread loads one element of each sub-matrix  
        if (tidr + bidr < M && tidc + i < N) {
            if (pre_a == 1) {
                As[tidr][tidc] = a[N * (tidr + bidr) + tidc + i];
            } else {
                As[tidr][tidc] = a[pre * N * M + N * (tidr + bidr) + tidc + i];
            }
        } else {
            As[tidr][tidc] = 0;
        }

        if (tidr + i < N && tidc + bidc < P) {
            if (pre_b == 1) {
                Bs[tidr][tidc] = b[P * (tidr + i) + tidc + bidc];
            } else {
                Bs[tidr][tidc] = b[pre * N * P + P * (tidr + i) + tidc + bidc];
            }
        } else {
            Bs[tidr][tidc] = 0;
        }

        // Synchronize to make sure the sub-matrices are loaded  
        // before starting the computation 
        __syncthreads();

        // Multiply Asub and Bsub together 
        for (int k = 0; k < TILE; ++k) {
            Outvalue += As[tidr][k] * Bs[k][tidc];
        }

        // Synchronize to make sure that the preceding computation is done before  
        // loading two new sub-matrices of A and B in the next iteration 
        __syncthreads();
    }

    // Write Outvalue to device memory  
    // Each thread writes one element 
    if (tidr + bidr < M && tidc + bidc < P) {
        out[pre * M * P + P * (tidr + bidr) + tidc + bidc] = Outvalue;
    }
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, 
        uint32_t pre_a, uint32_t pre_b, uint32_t M, uint32_t N, uint32_t P) {
    /**
     * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
     * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
     * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
     * over (i,j) entries in the output array.  However, to really get the full benefit of this
     * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
     * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
     * the CPU backend, here you should implement a single function that works across all size
     * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
     * implementations, this function here will largely just set up the kernel call, and you should
     * implement the logic in a separate MatmulKernel() call.
     * 
     *
     * Args:
     *   a: compact 2D array of size m x n
     *   b: comapct 2D array of size n x p
     *   out: compact 2D array of size m x p to write the output to
     *   M: rows of a / out
     *   N: columns of a / rows of b
     *   P: columns of b / out
     */
  
    uint32_t pre_max = pre_a;
    if (pre_b > pre_a) {
        pre_max = pre_b;
    }
    dim3 blockDim(TILE, TILE);
    dim3 gridDim((M + TILE - 1) / blockDim.x, (P + TILE - 1) / blockDim.y, pre_max);
    MatmulTileKernel<<<gridDim, blockDim>>>(a.ptr, b.ptr, out->ptr, pre_a, pre_b, M, N, P);
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////
__global__ void ReduceMaxKernel(scalar_t* a, scalar_t* out, size_t out_size, size_t reduce_size) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    scalar_t max = -0x3f3f3f3f;

    if (tid < out_size) {
        for (int i = tid * reduce_size; i < (tid + 1) * reduce_size; i++) max = MAX(max, a[i]);
        __syncthreads();

        // printf("max: %.3f\n", max);
        out[tid] = max;
    }
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
    /**
     * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
     * for simplicity you can perform each reduction in a single CUDA thread.
     * 
     * Args:
     *   a: compact array of size a.size = out.size * reduce_size to reduce over
     *   out: compact array to write into
     *   redice_size: size of the dimension to reduce over
     */
    CudaDims dim = CudaOneDim(out->size);
    ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
}


__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t out_size, size_t reduce_size) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    scalar_t sum = 0;

    if (tid < out_size) {
        for (int i = tid * reduce_size; i < (tid + 1) * reduce_size; i++) sum+=a[i];
        __syncthreads();

        // printf("sum: %.3f\n", sum);
        out[tid] = sum;
    }
}


void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
    /**
     * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
     * can perform each reduction in a single CUDA thread.
     * 
     * Args:
     *   a: compact array of size a.size = out.size * reduce_size to reduce over
     *   out: compact array to write into
     *   reduce_size: size of the dimension to reduce over
     */
    CudaDims dim = CudaOneDim(out->size);
    ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
    namespace py = pybind11;
    using namespace needle;
    using namespace cuda;

    m.attr("__device_name__") = "cuda";
    m.attr("__tile_size__") = TILE;

    py::class_<CudaArray>(m, "Array")
        .def(py::init<size_t>(), py::return_value_policy::take_ownership)
        .def_readonly("size", &CudaArray::size)
        .def("ptr", &CudaArray::ptr_as_int);

    // return numpy array, copying from CPU
    m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides, size_t offset) {
            std::vector<size_t> numpy_strides = strides;
            std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                    [](size_t& c) { return c * ELEM_SIZE; });
            // copy memory to host
            scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
            if (host_ptr == 0) throw std::bad_alloc();
            cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

            // return numpy array
            py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
            return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
    });

    // copy numpy array to GPU
    m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
        cudaError_t err =
            cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
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
    
    m.def("ewise_log", EwiseLog);
    m.def("ewise_exp", EwiseExp);
    m.def("ewise_tanh", EwiseTanh);
    
    m.def("matmul", Matmul);
    
    m.def("reduce_max", ReduceMax);
    m.def("reduce_sum", ReduceSum);
}

