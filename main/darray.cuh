#ifndef DEVICE_ARRAY_CLASS_H
#define DEVICE_ARRAY_CLASS_H

#include "cuda_utils.h"
#include "dmemory.cuh"
#include <vector>

template <typename T>
__global__ void _destruct_objs(T* objs_ptr, size_t length) {
	int gid = threadIdx.x * blockIdx.x * blockDim.x;
	if (gid >= length) return;

	objs_ptr[gid].~T();
}

template <typename T, bool destructObj = false>
class darray {
	dmemory dmem;

public:

	darray() = delete;
	darray(const darray&) = delete;
	darray& operator=(const darray&) = delete;

	darray(darray&& other) noexcept : dmem(std::move(other.dmem)) {}
	darray& operator=(darray&& other) noexcept { dmem = std::move(other.dmem); return *this; }

	darray(size_t size) : dmem(size * sizeof(T)) {}
	darray(const T* arr, size_t size) : darray(size * sizeof(T)) {
		CUDA_ASSERT(cudaMemcpy(dmem.getPtr(), arr, size * sizeof(T), cudaMemcpyHostToDevice));
	}
	darray(const std::vector<T>& v) : darray(v.data(), v.size()) {}

	T* getPtr() noexcept { return dmem.getPtr<T>(); }
	const T* getPtr() const noexcept { return dmem.getPtr<T>(); }

	T* operator=(size_t i) noexcept { return getPtr() + i; }
	const T* operator=(size_t i) const noexcept { return getPtr() + i; }
};

#endif // DEVICE_ARRAY_CLASS_H //