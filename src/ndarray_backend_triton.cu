#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

class CudaArray {
public:
    CudaArray(size_t size) : size(size) {
        cudaMalloc(&d_data, size * sizeof(float));
    }

    ~CudaArray() {
        cudaFree(d_data);
    }

    void copyFromHost(float* hostData) {
        cudaError_t err = cudaMemcpy(d_data, hostData, size * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memcpy H2D failed");
        }

    }

    void copyToHost(float* hostData) {
        cudaError_t err = cudaMemcpy(hostData, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memcpy D2H failed");
        }
    }

    float* ptr() { 
        return d_data; 
    }
  


    size_t getSize() const {
        return size;
    }

    float* d_data;
    size_t size;
};

PYBIND11_MODULE(ndarray_backend_triton, m) {
    py::class_<CudaArray>(m, "CudaArray")
        .def(py::init<size_t>())
        .def("copy_from_host", [](CudaArray &self, py::array_t<float> array) {
            self.copyFromHost(static_cast<float *>(array.request().ptr));
        })
        .def("copy_to_host", [](CudaArray &self) {
            py::array_t<float> array(self.size);
            self.copyToHost(static_cast<float *>(array.request().ptr));
            return array;
        })
        .def("ptr", [](CudaArray &self) {
            // 返回设备指针作为整数类型（uintptr_t），确保Python端的安全性
            return reinterpret_cast<uintptr_t>(self.ptr());
        })
        .def("size", &CudaArray::getSize)
        .def("dtype", &CudaArray::getSize);
}
