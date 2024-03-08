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

template <typename T, bool destruct = false>
class darray {
	dmemory dmem;
	size_t length;

public:

	darray() = delete;
	darray(const darray&) = delete;
	darray& operator=(const darray&) = delete;

	darray(darray&& other) noexcept : length(other.length), dmem(std::move(other.dmem)) {}
	darray& operator=(darray&& other) noexcept { length = other.length; dmem = std::move(other.dmem); return *this; }

	darray(size_t length) : length(length), dmem(length * sizeof(T)) {}
	darray(const T* arr, size_t length) : length(length), dmem(length * sizeof(T)) {
		CUDA_ASSERT(cudaMemcpy(dmem.getPtr(), arr, length * sizeof(T), cudaMemcpyHostToDevice));
	}
	darray(const std::vector<T>& v) : darray(v.data(), v.size()) {}

	~darray() {
		if (destruct) {
			int threads = 32;
			int blocks = ceilDiv(length, threads);
			_destruct_objs<T><<<blocks, threads>>>(getPtr(), length);
			CUDA_ASSERT(cudaDeviceSynchronize());
		}
	}

	size_t getLength() const { return length; }

	T* getPtr() noexcept { return dmem.getPtr<T>(); }
	const T* getPtr() const noexcept { return dmem.getPtr<T>(); }

	T* operator[](size_t i) noexcept { return getPtr() + i; }
	const T* operator[](size_t i) const noexcept { return getPtr() + i; }
};

#endif // DEVICE_ARRAY_CLASS_H //