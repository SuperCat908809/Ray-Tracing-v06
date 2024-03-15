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


template <class _cuManager_t, typename T, typename... Args>
concept cuManager_c = requires (_cuManager_t& a, T* ptr, Args&&... args) {
	{ a.make_at(ptr, args...) } -> std::same_as<T*>;
	{ a.destruct(ptr) };
};

template <class _cuManager_t, typename T, typename... Args>
concept cuManager_constructor_c = requires (_cuManager_t& a, T* ptr, Args&&... args) {
	{ a.construct_at(ptr, std::forward<Args>(args)...) };
};

template <class _cuManager_t, typename T>
concept cuManager_destructor_c = requires (_cuManager_t& a, T* ptr) {
	{ a.destruct(ptr) };
};

template <class _cuManager_t, typename T>
concept cuManager_move_c = requires (_cuManager_t& a, T* dst, T* src) {
	{ a.move_to(dst, src) };
};

template <class _cuManager_t, typename T>
concept cuManager_copy_c = requires (_cuManager_t& a, T* dst, T* src) {
	{ a.copy_to(dst, src) };
}

#endif // CUDA_ALLOCATOR_ABSTRACT_CUH //