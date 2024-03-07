#ifndef D_ABSTRACT_CLASSES_INL
#define D_ABSTRACT_CLASSES_INL

#include "dAbstracts.cuh"


template <typename T, typename U, typename... Args> requires std::derived_from<U, T>
__global__ inline void _makeArrayOnDevice(size_t count, T** ptrs, Args*... args) {
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gid >= count) return;

	ptrs[gid] = new U(args[gid]...);
}

template <typename T, typename FactoryType>
__global__ inline void _makeArrayOnDeviceFactory(size_t count, size_t input_offset, T** ptrs, FactoryType* f) {
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gid >= count) return;

	ptrs[gid] = f->operator()(gid + input_offset);
}

template <typename T>
__global__ inline void _deleteArrayOnDevice(size_t count, T** ptrs) {
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gid >= count) return;

	if (ptrs[gid]) {
		delete ptrs[gid];
		ptrs[gid] = nullptr;
	}
}


template <typename T>
void dAbstractArray<T>::_delete() {
	DeleteOnDevice(length, 0);
	//CUDA_ASSERT(cudaFree(ptr2));
	//length = 0;
	//ptr2 = nullptr;
}

template <typename T>
dAbstractArray<T>::dAbstractArray(size_t size) : ptrs(size) {
	length = size;
	//CUDA_ASSERT(cudaMalloc((void**)&ptr2, sizeof(T*) * length));
}

template <typename T>
dAbstractArray<T>::~dAbstractArray() {
	_delete();
}

template <typename T>
template <typename U> requires std::derived_from<U, T>
dAbstractArray<T>::dAbstractArray(dAbstractArray<U>&& other) : 
	ptrs(std::move(other.ptrs)) {
	length = other.length;
	//ptr2 = other.ptr2;

	other.length = 0ull;
	//other.ptr2 = nullptr;
}

template <typename T>
template <typename U> requires std::derived_from<U, T>
dAbstractArray<T>& dAbstractArray<T>::operator=(dAbstractArray<U>&& other) {
	_delete();

	ptrs = std::move(other.ptrs);

	length = other.length;
	//ptr2 = other.ptr2;

	other.length = 0ull;
	//other.ptr2 = nullptr;

	return *this;
}



template <typename T>
std::vector<T*> dAbstractArray<T>::getPtrVector() {
	std::vector<T*> vec(length);
	CUDA_ASSERT(cudaMemcpy(vec.data(), ptrs.getPtr(), sizeof(T*) * length, cudaMemcpyDeviceToHost));
	return vec;
}

template <typename T>
std::vector<const T*> dAbstractArray<T>::getPtrVector() const {
	std::vector<const T*> vec(length);
	CUDA_ASSERT(cudaMemcpy(vec.data(), ptrs.getPtr(), sizeof(T*) * length, cudaMemcpyDeviceToHost));
	return vec;
}


template <typename T>
template <typename U, typename... Args> requires std::derived_from<U, T>
void dAbstractArray<T>::MakeOnDevice(size_t count, size_t offset, size_t input_offset, const Args*... args) {
	int threads = 32;
	int blocks = ceilDiv(count, threads);
	_makeArrayOnDevice<T, U><<<blocks, threads>>>(count, ptrs.getPtr() + offset, (args + input_offset)...);
	CUDA_ASSERT(cudaDeviceSynchronize());
}

template <typename T>
template <typename DeviceFactoryType>
void dAbstractArray<T>::MakeOnDeviceFactory(size_t count, size_t offset, size_t input_offset, DeviceFactoryType* factory) {
	int threads = 32;
	int blocks = ceilDiv(count, threads);
	_makeArrayOnDeviceFactory<T, DeviceFactoryType><<<blocks, threads>>>(count, input_offset, ptrs.getPtr() + offset, factory);
	CUDA_ASSERT(cudaDeviceSynchronize());
}


template <typename T>
void dAbstractArray<T>::DeleteOnDevice(size_t count, size_t offset) {
	int threads = 32;
	int blocks = ceilDiv(count, threads);
	_deleteArrayOnDevice<T><<<blocks, threads>>>(count, ptrs.getPtr() + offset);
	CUDA_ASSERT(cudaDeviceSynchronize());
}

#endif // D_ABSTRACT_CLASSES_INL //