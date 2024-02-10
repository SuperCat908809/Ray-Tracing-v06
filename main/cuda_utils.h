#ifndef CUDA_UTILITIES_H
#define CUDA_UTILITIES_H

#include <assert.h>
#include <stdlib.h>
#include <stdexcept>
#include <cuda_runtime.h>

#define CUDA_CHECK(func) cudaAssert(func, #func, __FILE__, __LINE__)
#define CUDA_ASSERT(func) try { CUDA_CHECK(func); } catch (const std::runtime_error& e) { assert(0); }
inline void cudaAssert(cudaError_t code, const char* func, const char* file, const int line) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPU assert: %s %s\n%s %d\n%s :: %s",
			cudaGetErrorName(code), func,
			file, line,
			cudaGetErrorName(code), cudaGetErrorString(code)
		);
		throw std::runtime_error(cudaGetErrorString(code));
	}
}


template <typename T, typename... Args>
__global__ inline void handledMakeOnDevice(T** ptr2, Args... args) {
	if (!(threadIdx.x == 0 && blockIdx.x == 0)) return;

	(*ptr2) = new T(args...);
}

template <typename T>
__global__ inline void handledDeleteOnDevice(T** ptr2) {
	if (!(threadIdx.x == 0 && blockIdx.x == 0)) return;

	delete* ptr2;
}

template <typename T>
struct HandledDeviceAbstract {
private:
	T* ptr{};
	T** ptr2{};

public:

	template <typename... Args>
	__host__ HandledDeviceAbstract(Args... args) {
		CUDA_ASSERT(cudaMalloc(&ptr2, sizeof(T*)));
		handledMakeOnDevice << <1, 1 >> > (ptr2, args...);
		CUDA_ASSERT(cudaPeekAtLastError());
		CUDA_ASSERT(cudaMemcpy(&ptr, ptr2, sizeof(T*), cudaMemcpyDeviceToHost));
	}
	__host__ ~HandledDeviceAbstract() {
		handledDeleteOnDevice << <1, 1 >> > (ptr2);
		CUDA_ASSERT(cudaPeekAtLastError());
		CUDA_ASSERT(cudaDeviceSynchronize());
		CUDA_ASSERT(cudaGetLastError());
		CUDA_ASSERT(cudaFree(ptr2));
	}

	T* getPtr() const { return ptr; }
};

#endif // CUDA_UTILITIES_H //