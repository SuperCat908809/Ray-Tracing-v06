#ifndef CUDA_ALLOCATOR_ABSTRACT_CUH
#define CUDA_ALLOCATOR_ABSTRACT_CUH

#include <inttypes.h>
#include <cuda_runtime.h>
#include <concepts>


template <class T> requires std::is_object_v<T>
class cuObjManager {
public:

	template <typename... Args> requires std::constructible_from<T, Args...>
	__device__ void construct_at(T* ptr, Args&&... args) const { 
		new (ptr) (std::forward<Args>(args)...);
	}
	__device__ void move_to(T* dst, T&& src) const requires std::is_move_assignable<T> { 
		*dst = std::forward<T>(src);
	}
	__device__ void move_to_uninitialized(T* dst, T&& src) const requires std::is_move_constructible<T> { 
		new (dst) T(std::forward<T>(src));
	}
	__device__ void copy_to(T* dst, const T& src) const requires std::is_copy_assignable<T> { 
		*dst = src;
	}
	__device__ void copy_to_uninitialized(T* dst, const T& src) const requires std::is_copy_constructible<T> { 
		new (dst) T(src);
	}
	__device__ void destruct(T* obj) const requires std::is_destructible<T> {
		obj->~T();
	}
};

template <class T, bool responsible>
class cuPtrManager {
public:

	template <class U> requires std::derived_from<U, T> && std::is_default_constructible<U>
	__device__ void default_initialize(T** ptr) const {
		*ptr = new U();
	}
	template <class U, typename... Args> requires std::derived_from<U, T> && std::constructible_from<U, Args...>
	__device__ void construct_at(T** ptr, Args&&... args) const {
		if constexpr (responsible) if (*ptr) delete* ptr;
		*ptr = new U(std::forward<Args>(args)...);
	}
	template <class U> requires std::derived_from<U, T> && std::is_copy_constructible<U>
	__device__ void copy_to(T** dst, const U& src) const {
		if constexpr (responsible) if (*dst) delete* dst;
		*dst = new U(src);
	}
	template <class U> requires std::derived_from<U, T> && std::is_copy_constructible<U>
	__device__ void copy_to_uninitialized(T** dst, const U& src) const {
		*dst = new U(src);
	}
	template <class U> requires std::derived_from<U, T> && std::is_move_constructible<U>
	__device__ void move_to(T** dst, U&& src) const {
		if constexpr (responsible) if (*dst) delete* dst;
		*dst = new U(std::forward<U>(src));
	}
	template <class U> requires std::derived_from<U, T> && std::is_move_constructible<U>
	__device__ void move_to_uninitialized(T** dst, U&& src) const {
		*dst = new U(std::forward<U>(src));
	}
	__device__ void destruct(T** ptr) const {
		if constexpr (responsible) if (*ptr) delete* ptr;
		*ptr = nullptr;
	}
};

#endif // CUDA_ALLOCATOR_ABSTRACT_CUH //