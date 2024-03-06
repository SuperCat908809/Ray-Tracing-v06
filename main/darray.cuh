#ifndef DEVICE_ARRAY_CLASS_H
#define DEVICE_ARRAY_CLASS_H

#include "cuda_utils.h"
#include "dmemory.cuh"
#include <vector>

template <typename T>
class darray {
	dmemory dmem;

public:

	darray() = delete;
	darray(const darray&) = delete;

	__host__ darray(darray&& other) noexcept : dmem(std::move(other.dmem)) {}
	__host__ darray& operator=(darray&& other) noexcept { dmem = std::move(other.dmem) return *this; }

	__host__ darray(size_t size) : dmem(size * sizeof(T)) {}
	__host__ darray(const std::vector<const T>& v) : darray(v.data(), v.size()) {}
	__host__ darray(const T* arr, size_t size) : darray(size * sizeof(T)) {
		CUDA_ASSERT(cudaMemcpy(dmem.getPtr(), arr, size * sizeof(T), cudaMemcpyHostToDevice));
	}

	__host__ __device__ T* getPtr() noexcept { return dmem.getPtr<T>(); }
	__host__ __device__ const T* getPtr() const noexcept { return dmem.getPtr<T>(); }

	T& operator=(size_t i) noexcept { return getPtr()[i]; }
	const T& operator=(size_t i) const noexcept { return getPtr()[i]; }
};

#endif // DEVICE_ARRAY_CLASS_H //