#ifndef CUDA_UTILITIES_CUH
#define CUDA_UTILITIES_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <concepts>
#include "cuError.h"

template <typename T, typename... Args> requires std::constructible_from<T, Args...>
__global__ inline void _makeOnDeviceKer(T* dst_T, Args... args) {
	if (threadIdx.x != 0 || blockIdx.x != 0) return;

	new (dst_T) T(args...);
}

template <typename T, typename... Args> requires std::constructible_from<T, Args...>
inline T* newOnDevice(const Args&... args) {
	T* ptr = nullptr;
	CUDA_ASSERT(cudaMalloc((void**)&ptr, sizeof(T)));
	_makeOnDeviceKer<T, Args...><<<1, 1>>>(ptr, args...);
	CUDA_ASSERT(cudaDeviceSynchronize());
	return ptr;
}

template <std::movable T>
__device__ inline constexpr void cuda_swap(T& a, T& b) {
	T tmp = std::move(a);
	a = std::move(b);
	b = std::move(tmp);
}


#endif // CUDA_UTILITIES_CUH //