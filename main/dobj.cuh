#ifndef DEVICE_OBJECT_CLASS_H
#define DEVICE_OBJECT_CLASS_H

#include "cuda_utils.h"
#include "dmemory.cuh"

template <typename T>
class dobj {
	dmemory dmem;

public:

	dobj() = delete;
	dobj(const dobj&) = delete;

	__host__ dobj(dobj&& other) noexcept : dmem(std::move(other.dmem)) {}
	__host__ dobj& operator=(dobj&& other) noexcept { dmem = std::move(other.dmem); return *this; }

	__host__ dobj(T obj) : dmem(sizeof(T)) {
		CUDA_ASSERT(cudaMemcpy(dmem.getPtr(), &obj, sizeof(T), cudaMemcpyHostToDevice));
	}

	__host__ __device__ T* getPtr() { return d_obj; }
	__host__ __device__ const T* getPtr() const { return d_obj; }
};

#endif // DEVICE_OBJECT_CLASS_H //