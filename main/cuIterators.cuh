#ifndef CUDA_ITERATOR_CLASSES_AND_CONCEPTS_CUH
#define CUDA_ITERATOR_CLASSES_AND_CONCEPTS_CUH

#include <inttypes.h>
#include <cuda_runtime.h>
#include <concepts>


enum cuIteratorCategory { INPUT, FORWARD, BIDIRECTIONAL, RANDOM_ACCESS };
template <cuIteratorCategory category, class T>
class cuIterator {};

template <cuIteratorCategory category, class T>
class cuInputIterator : public cuIterator<category, T> {
public:
	__host__ __device__ cuInputIterator() = default;
	__host__ __device__ cuInputIterator(const cuInputIterator&) = default;
	__host__ __device__ cuInputIterator& operator=(const cuInputIterator&) = default;

	__host__ __device__ bool operator==(const cuInputIterator&) const = 0;
	__host__ __device__ bool operator!=(const cuInputIterator&) const = 0;

	__host__ __device__ T&& operator*() const = 0;
	__host__ __device__ T&& operator->() const = 0;

	__host__ __device__ void operator++() const = 0;
};



template <typename T>
class cuRandomAccessIterator {
	T* ptr{ nullptr };
public:

	__host__ __device__ cuRandomAccessIterator() = default;
	__host__ __device__ cuRandomAccessIterator(T* ptr) : ptr(ptr) {}
	__host__ __device__ cuRandomAccessIterator(const cuRandomAccessIterator&) = default;
	cuRandomAccessIterator& operator=(const cuRandomAccessIterator&) = default;

	__host__ __device__ operator==(const cuRandomAccessIterator& other) const { return ptr == other.ptr; }
	__host__ __device__ operator!=(const cuRandomAccessIterator& other) const { return ptr != other.ptr; }

	//__host__ __device__ 
};

#endif // CUDA_ITERATOR_CLASSES_AND_CONCEPTS_CUH //