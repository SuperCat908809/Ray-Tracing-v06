#ifndef D_ABSTRACT_CLASSES_INL
#define D_ABSTRACT_CLASSES_INL

#include "dAbstracts.cuh"

#include <inttypes.h>
#include <cuda_runtime.h>
#include <concepts>
#include <vector>
#include "cuError.h"

#include "ceilDiv.h"


template <typename T, typename U, typename... Args> requires std::derived_from<U, T>
__global__ inline void _makeArrayOnDevice(T** ptrs, size_t count, Args*... args) {
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gid >= count) return;

	ptrs[gid] = new U(args[gid]...);
}

template <typename T, typename FactoryType>
__global__ inline void _makeArrayOnDeviceFactory(T** ptrs, size_t count, size_t input_offset, FactoryType* f) {
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gid >= count) return;

	ptrs[gid] = f->operator()(gid + input_offset);
}

template <typename T>
__global__ inline void _deleteArrayOnDevice(T** ptrs, size_t count) {
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gid >= count) return;

	if (ptrs[gid]) {
		delete ptrs[gid];
		ptrs[gid] = nullptr;
	}
}


template <typename T, bool d>
void dAbstractArray<T, d>::_delete() {
	if (d)
		DeleteOnDevice(getLength(), 0);
}
}

template <typename T, bool d>
template <typename U> requires std::derived_from<U, T>
dAbstractArray<T, d>::dAbstractArray(dAbstractArray<U, d>&& other) : ptrs(std::move(other.ptrs)) {}

template <typename T, bool d>
template <typename U> requires std::derived_from<U, T>
dAbstractArray<T, d>& dAbstractArray<T, d>::operator=(dAbstractArray<U, d>&& other) {
	_delete();

	ptrs = std::move(other.ptrs);

	return *this;
}



template <typename T, bool d>
std::vector<T*> dAbstractArray<T, d>::getPtrVector() {
	std::vector<T*> vec;
	vec.resize(getLength());
	CUDA_ASSERT(cudaMemcpy(vec.data(), ptrs.getPtr(), sizeof(T*) * getLength(), cudaMemcpyDeviceToHost));
	return vec;
}

template <typename T, bool d>
std::vector<const T*> dAbstractArray<T, d>::getPtrVector() const {
	std::vector<const T*> vec;
	vec.resize(getLength());
	CUDA_ASSERT(cudaMemcpy(vec.data(), ptrs.getPtr(), sizeof(T*) * getLength(), cudaMemcpyDeviceToHost));
	return vec;
}


template <typename T, bool d>
template <typename U, typename... Args> requires std::derived_from<U, T>
void dAbstractArray<T, d>::MakeOnDevice(size_t count, size_t offset, size_t input_offset, const Args*... args) {
	int threads = 32;
	int blocks = ceilDiv(count, threads);
	_makeArrayOnDevice<T, U><<<blocks, threads>>>(ptrs.getPtr() + offset, count, (args + input_offset)...);
	CUDA_ASSERT(cudaDeviceSynchronize());
}

template <typename T, bool d>
template <typename DeviceFactoryType>
void dAbstractArray<T, d>::MakeOnDeviceFactory(size_t count, size_t offset, size_t input_offset, DeviceFactoryType* factory) {
	int threads = 32;
	int blocks = ceilDiv(count, threads);
	_makeArrayOnDeviceFactory<T, DeviceFactoryType><<<blocks, threads>>>(ptrs.getPtr() + offset, count, input_offset, factory);
	CUDA_ASSERT(cudaDeviceSynchronize());
}


template <typename T, bool d>
void dAbstractArray<T, d>::DeleteOnDevice(size_t count, size_t offset) {
	int threads = 32;
	int blocks = ceilDiv(count, threads);
	_deleteArrayOnDevice<T><<<blocks, threads>>>(ptrs.getPtr() + offset, count);
	CUDA_ASSERT(cudaDeviceSynchronize());
}

#endif // D_ABSTRACT_CLASSES_INL //