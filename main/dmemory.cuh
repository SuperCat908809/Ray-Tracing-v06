#ifndef DEVICE_MEMORY_CLASS_H
#define DEVICE_MEMORY_CLASS_H

#include <inttypes.h>
#include <cuda_runtime.h>
#include "cuError.h"


class dmemory {
	void* device_memory;

	void _free() {
		if (device_memory)
			CUDA_ASSERT(cudaFree(device_memory));
	}

public:

	dmemory() = delete;
	dmemory(const dmemory&) = delete;
	dmemory& operator=(const dmemory&) = delete;

	dmemory(dmemory&& other) noexcept {
		device_memory = other.device_memory;
		other.device_memory = nullptr;
	}
	dmemory& operator=(dmemory&& other) noexcept {
		_free();

		device_memory = other.device_memory;
		other.device_memory = nullptr;

		return *this;
	}

	dmemory(size_t size) {
		CUDA_ASSERT(cudaMalloc(&device_memory, size));
	}
	~dmemory() {
		_free();
	}

	const void* getPtr() const noexcept { return device_memory; }
	void* getPtr() noexcept { return device_memory; }

	template <typename T> const T* getPtr() const noexcept { return static_cast<const T*>(getPtr()); }
	template <typename T> T* getPtr() noexcept { return static_cast<T*>(getPtr()); }
};

#endif // DEVICE_MEMORY_CLASS_H //