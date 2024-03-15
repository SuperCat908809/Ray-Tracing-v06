#ifndef CUDA_ALLOCATOR_ABSTRACT_CUH
#define CUDA_ALLOCATOR_ABSTRACT_CUH

#include <inttypes.h>
#include <cuda_runtime.h>
#include <concepts>


template <class _cuAlloc_t, typename T>
concept cuAllocator = requires (_cuAlloc_t&& a, _cuAlloc_t && b, size_t count, T* to_be_deallocated) {
	{ a.allocate(count) } -> std::same_as<T>;
	{ a.deallocate(to_be_deallocated) };
};

#endif // CUDA_ALLOCATOR_ABSTRACT_CUH //