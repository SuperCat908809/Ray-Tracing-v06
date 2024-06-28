#ifndef DEVICE_POINTER_ARRAY_CLASS_CUH
#define DEVICE_POINTER_ARRAY_CLASS_CUH

#include <inttypes.h>
#include <cuda_runtime.h>
#include "../cuError.h"
#include "dmemory.cuh"
#include "dobj.cuh"
#include <concepts>
#include <vector>


#ifdef __CUDACC__
#include "ceilDiv.h"
#include <device_launch_parameters.h>

template <typename T>
__global__ void _dptrarray_destruct_kernel(T** arr, size_t length) {
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gid >= length) return;

	T*& ptr = arr[gid];
	if (ptr) {
		delete ptr;
		ptr = nullptr;
	}
}

#endif


template <class T, bool responsible>
class dptrarray {
	dmemory dmem;
	size_t length;

	void _destruct() {
	#ifndef __CUDACC__
		static_assert(!(responsible), "Cannot launch destructor kernel without NVCC compilation.");
	#else
		if constexpr (responsible) {
			if (getPtr()) {
				int threads = 32;
				int blocks = ceilDiv(length, threads);
				_dptrarray_destruct_kernel<T><<<blocks, threads>>>(getPtr(), length);
				CUDA_ASSERT(cudaDeviceSynchronize());
			}
		}
	#endif
	}


public:

	dptrarray() = delete;
	dptrarray(const dptrarray&) = delete;
	dptrarray& operator=(const dptrarray&) = delete;

	dptrarray(dptrarray&& other) noexcept : length(other.length), dmem(std::move(other.dmem)) {}
	dptrarray& operator=(dptrarray&& other) noexcept {
		_destruct();
		length = other.length;
		dmem = std::move(other.dmem);
		return *this;
	}

	enum MemorySource { FROM_HOST = cudaMemcpyHostToDevice, FROM_DEVICE = cudaMemcpyDeviceToDevice };

	dptrarray(size_t length) : length(length), dmem(length * sizeof(T*)) {}
	dptrarray(const T** arr, size_t length, MemorySource source = FROM_HOST) : length(length), dmem(length * sizeof(T*)) {
		CUDA_ASSERT(cudaMemcpy(dmem.getPtr(), arr, length * sizeof(T*), (cudaMemcpyKind)source));
	}
	dptrarray(const std::vector<T*>& v) : darray(v.data(), v.size()) {}
#if 0
	template <bool d>
	dptrarray(std::vector<dobj<T, d>>&& v) : length(v.size()), dmem(v.size() * sizeof(T*)) {
		std::vector<T*> v2(v.size());
		for (int i = 0; i < v.size(); i++) {
			v2.push_back(v[i].transfer_ownership());
		}
		CUDA_ASSERT(cudaMemcpy(dmem.getPtr(), v2.data(), v2.size() * sizeof(T*), cudaMemcpyHostToDevice));
	}
#endif

	size_t getLength() const { return length; }

	T** getPtr() noexcept { return dmem.getPtr<T**>(); }
	const T** getPtr() const noexcept { return dmem.getPtr<T**>(); }

	T** operator[](size_t i) noexcept { return getPtr() + i; }
	const T** operator[](size_t i) const noexcept { return getPtr() + i; }
};

#endif // DEVICE_POINTER_ARRAY_CLASS_CUH //