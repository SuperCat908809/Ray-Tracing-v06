#ifndef CUDA_ITERATOR_CLASSES_AND_CONCEPTS_CUH
#define CUDA_ITERATOR_CLASSES_AND_CONCEPTS_CUH

#include <inttypes.h>
#include <cuda_runtime.h>
//#include <concepts>


// i could revisit this in the future but for now i will stick with my own iterator class and isolate components like this when necessary
#if 0
enum cuIteratorCategory { INPUT, OUTPUT, FORWARD, BIDIRECTIONAL, RANDOM_ACCESS };
template <cuIteratorCategory category, class T>
class cuIterator {
public:
	__host__ __device__ cuIterator(const cuIterator&) = default;
	__host__ __device__ cuIterator& operator=(const cuIterator&) = default;

	__host__ __device__ void operator++() const = 0;
};


template <cuIteratorCategory category, class T>
class cuInputIterator : public cuIterator<category, T> {
public:
	__host__ __device__ bool operator==(const cuInputIterator&) const = 0;
	__host__ __device__ bool operator!=(const cuInputIterator&) const = 0;

	__host__ __device__ T&& operator*() const = 0;
	__host__ __device__ T&& operator->() const = 0;
};

template <cuIteratorCategory category, class T>
class cuOutputIterator : public cuIterator<category, T> {
	__host__ __device__ bool operator==(const cuInputIterator&) const = 0;
	__host__ __device__ bool operator!=(const cuInputIterator&) const = 0;

	__host__ __device__ T& operator*() = 0;
	__host__ __device__ T& operator->() = 0;
};


template <cuIteratorCategory category, class T>
class cuForwardIterator : public cuInputIterator<category, T> {

};
#endif

template <class T>
class cuIterator {
public:

	__device__ cuIterator() = default;
	__device__ cuIterator(const cuIterator&) = default;
	__device__ cuIterator& operator=(const cuIterator&) = default;

	__device__ cuIterator& operator++() = 0;

	__device__ T operator*() const = 0;
	__device__ T& operator*() = 0;
	__device__ const T* operator->() const = 0;
	__device__ T* operator->() = 0;

	__device__ T operator[](size_t) const = 0;
	__device__ T& operator[](size_t) = 0;

	__device__ bool operator==(const cuIterator&) const = 0;
	__device__ bool operator!=(const cuIterator&) const = 0;
	__device__ bool operator<(const cuIterator&) const = 0;
	__device__ bool operator>(const cuIterator&) const = 0;
	__device__ bool operator<=(const cuIterator&) const = 0;
	__device__ bool operator>=(const cuIterator&) const = 0;
};


template <class T>
class cuPtrIterator : public cuIterator<T> {
	T* ptr{ nullptr };
public:

	__device__ cuPtrIterator() = default;
	__device__ cuPtrIterator(T* ptr) : ptr(ptr) {}

	__device__ cuPtrIterator& operator++() { ptr++; return *this; }

	__device__ T operator*() const { return *ptr; }
	__device__ T& operator*() { return *ptr; }
	__device__ const T* operator->() const { return ptr; }
	__device__ T* operator->() { return ptr; }

	__device__ T operator[](size_t i) const { return ptr[i]; }
	__device__ T& operator[](size_t i) const { return ptr[i]; }

	__device__ bool operator==(const cuPtrIterator& other) const { return ptr == other.ptr; }
	__device__ bool operator!=(const cuPtrIterator& other) const { return ptr != other.ptr; }
	__device__ bool operator<(const cuPtrIterator& other) const { return ptr < other.ptr; }
	__device__ bool operator>(const cuPtrIterator& other) const { return ptr > other.ptr; }
	__device__ bool operator<=(const cuPtrIterator& other) const { return ptr <= other.ptr; }
	__device__ bool operator>=(const cuPtrIterator& other) const { return ptr >= other.ptr; }
};

#endif // CUDA_ITERATOR_CLASSES_AND_CONCEPTS_CUH //