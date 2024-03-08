#ifndef DEVICE_OBJECT_CLASS_H
#define DEVICE_OBJECT_CLASS_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuError.h"
#include "dmemory.cuh"
#include <concepts>


template <typename T, typename... Args>
__global__ void _make_dobj(T* obj_ptr, Args... args) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		new (obj_ptr) T(args...);
	}
}

template <typename T>
__global__ void _destruct_dobj(T* obj_ptr) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		obj_ptr->~T();
	}
}

template <typename T, bool destruct = false>
class dobj {
	template <typename U, bool d>
	friend class dobj;

	dmemory dmem;

	void _destruct() {
	#if destruct
		if (getPtr()) {
			_destruct_dobj<T><<<1, 1>>>(getPtr());
			CUDA_ASSERT(cudaDeviceSynchronize());
		}
	#endif
	}

	dobj() : dmem(sizeof(T)) {}

public:

	dobj(const dobj&) = delete;
	dobj& operator=(const dobj&) = delete;

	template <typename U> requires std::derived_from<U, T>
	dobj(dobj<U, destruct>&& other) noexcept : dmem(std::move(other.dmem)) {}

	template <typename U> requires std::derived_from<U, T>
	dobj& operator=(dobj<U, destruct>&& other) noexcept { _destruct(); dmem = std::move(other.dmem); return *this; }

	~dobj() {
		_destruct();
	}

	template <typename... Args>
	static dobj Make(const Args&... args) {
		dobj device_obj;
		_make_dobj<T><<<1, 1>>>(device_obj.getPtr(), args...);
		CUDA_ASSERT(cudaDeviceSynchronize());
		return device_obj;
	}
	template <typename U> requires std::derived_from<U, T>
	static dobj<T> Copy(const U& obj) {
		dobj<U> device_obj;
		CUDA_ASSERT(cudaMemcpy(device_obj.getPtr(), &obj, sizeof(U), cudaMemcpyHostToDevice));
		return device_obj;
	}

	T* getPtr() { return dmem.getPtr<T>(); }
	const T* getPtr() const { return dmem.getPtr<T>(); }
};

#endif // DEVICE_OBJECT_CLASS_H //