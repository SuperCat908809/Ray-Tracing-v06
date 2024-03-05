#ifndef DEVICE_OBJECT_CLASS_H
#define DEVICE_OBJECT_CLASS_H

#include "cuda_utils.h"

template <typename T>
class dobj {
	T* d_obj{ nullptr };

public:

	dobj() = delete;
	dobj(const dobj&) = delete;

	dobj(dobj&& other) {
		d_obj = other.d_obj;
		other.d_obj = nullptr;
	}
	dobj& operator=(dobj&& other) {
		if (d_obj != nullptr)
			CUDA_ASSERT(cudaFree(d_obj));
		d_obj = other.d_obj;
		other.d_obj = nullptr;
		return *this;
	}

	__host__ dobj(T obj) {
		CUDA_ASSERT(cudaMalloc((void**)&d_obj, sizeof(T)));
		CUDA_ASSERT(cudaMemcpy(d_obj, &obj, sizeof(T), cudaMemcpyHostToDevice));
	}
	__host__ ~dobj() {
		if (d_obj != nullptr)
			CUDA_ASSERT(cudaFree(d_obj));
	}

	__host__ __device__ T* getPtr() { return d_obj; }
	__host__ __device__ const T* getPtr() const { return d_obj; }
};

#endif // DEVICE_OBJECT_CLASS_H //