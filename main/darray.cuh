#ifndef DEVICE_ARRAY_CLASS_H
#define DEVICE_ARRAY_CLASS_H

#include <inttypes.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuError.h"
#include "dmemory.cuh"
#include <vector>

#include "ceilDiv.h"


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

	void _destruct() {
		if (getPtr() && destruct) {
			int threads = 32;
			int blocks = ceilDiv(length, threads);
			_destruct_objs<T><<<blocks, threads>>>(getPtr(), length);
			CUDA_ASSERT(cudaDeviceSynchronize());
		}
	}

public:

	darray() = delete;
	darray(const darray&) = delete;
	darray& operator=(const darray&) = delete;

	darray(darray&& other) noexcept : length(other.length), dmem(std::move(other.dmem)) {}
	darray& operator=(darray&& other) noexcept { _destruct(); length = other.length; dmem = std::move(other.dmem); return *this; }

	enum MemorySource { FROM_HOST = cudaMemcpyHostToDevice, FROM_DEVICE = cudaMemcpyDeviceToDevice };
	darray(size_t length) : length(length), dmem(length * sizeof(T)) {}
	darray(const T* arr, size_t length, MemorySource source = FROM_HOST) : length(length), dmem(length * sizeof(T)) {
		CUDA_ASSERT(cudaMemcpy(dmem.getPtr(), arr, length * sizeof(T), (cudaMemcpyKind)source));
	}
	darray(const std::vector<T>& v) : darray(v.data(), v.size()) {}

	~darray() {
		_destruct();
	}

	size_t getLength() const { return length; }

	T* getPtr() noexcept { return dmem.getPtr<T>(); }
	const T* getPtr() const noexcept { return dmem.getPtr<T>(); }

	T* operator[](size_t i) noexcept { return getPtr() + i; }
	const T* operator[](size_t i) const noexcept { return getPtr() + i; }
};

#endif // DEVICE_ARRAY_CLASS_H //