#ifndef DEVICE_ARRAY_CLASS_H
#define DEVICE_ARRAY_CLASS_H

#include "cuda_utils.h"
#include <vector>

template <typename T>
class darray {
	T* data{ nullptr };

public:

	darray() = delete;
	darray(const darray&) = delete;

	darray(darray&& other) {
		data = other.data;
		other.data = nullptr;
	}
	darray& operator=(darray&& other) {
		if (data != nullptr)
			CUDA_ASSERT(cudaFree(data));
		data = other.data;
		other.data = nullptr;
		return *this;
	}

	__host__ darray(const std::vector<T>& v) {
		CUDA_ASSERT(cudaMalloc((void**)&data, sizeof(T) * v.size()));
		CUDA_ASSERT(cudaMemcpy(data, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice));
	}
	__host__ ~darray() {
		CUDA_ASSERT(cudaFree(data));
	}

	__host__ __device__ T* getPtr() { return data; }
	__host__ __device__ const T* getPtr() const { return data; }
};

#endif // DEVICE_ARRAY_CLASS_H //