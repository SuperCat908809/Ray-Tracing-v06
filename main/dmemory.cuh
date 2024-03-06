#ifndef DEVICE_MEMORY_CLASS_H
#define DEVICE_MEMORY_CLASS_H

#include "cuda_utils.h"

class dmemory {
	void* device_memory;

	__host__ void _free() {
		if (device_memory)
			CUDA_ASSERT(cudaFree(device_memory));
	}

public:

	dmemory() = delete;
	dmemory(const dmemory&) = delete;
	dmemory& operator=(const dmemory&) = delete;

	__host__ dmemory(dmemory&& other) noexcept {
		device_memory = other.device_memory;
		other.device_memory = nullptr;
	}
	__host__ dmemory& operator=(dmemory&& other) noexcept {
		_free();

		device_memory = other.device_memory;
		other.device_memory = nullptr;

		return *this;
	}

	template <typename T>
	__host__ dmemory(size_t size) : dmemory(sizeof(T) * size) {}
	__host__ dmemory(size_t size) {
		CUDA_ASSERT(cudaMalloc(&device_memory, size));
	}
	__host__ ~dmemory() {
		_free();
	}

	__host__ __device__ template <typename T> const T* getPtr() const noexcept { return static_cast<const T*>(getPtr()); }
	__host__ __device__ template <typename T> T* getPtr() noexcept { return static_cast<T*>(getPtr()); }

	__host__ __device__ const void* getPtr() const noexcept { return device_memory; }
	__host__ __device__ void* getPtr() noexcept { return device_memory; }
};

#endif // DEVICE_MEMORY_CLASS_H //