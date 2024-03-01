#ifndef HANDLED_DEVICE_ABSTRACT_CLASSES_INL
#define HANDLED_DEVICE_ABSTRACT_CLASSES_INL

#include "handled_device_abstracts.cuh"

template <typename T1, typename T2, typename... Args>
__global__ inline void _makeOnDevice(T1** ptr, Args... args) {
	if (!(threadIdx.x == 0 && blockIdx.x == 0)) return;

	(*ptr) = new T2(args...);
}

template <typename T>
__global__ inline void _deleteOnDevice(T** ptr) {
	if (!(threadIdx.x == 0 && blockIdx.x == 0)) return;

	if (*ptr == nullptr) return;
	delete* ptr;
	ptr = nullptr;
}


template <typename T1, typename T2, typename... Args>
__global__ inline void _makeArrayOnDevice(size_t count, T1** ptrs, Args*... args) {
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gid >= count) return;

	ptrs[gid] = new T2(args[gid]...);
}

template <typename T, typename FactoryType>
__global__ inline void _makeArrayOnDeviceFactory(size_t count, size_t input_offset, T** ptrs, FactoryType* f) {
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gid >= count) return;

	ptrs[gid] = f->operator()(gid + input_offset);
}
template <typename T, typename FactoryType>
__global__ inline void _makeArrayOnDeviceFactory(size_t count, size_t input_offset, T** ptrs, FactoryType f) {
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gid >= count) return;

	ptrs[gid] = f(gid + input_offset);
}

template <typename T>
__global__ inline void _deleteArrayOnDevice(size_t count, T** ptrs) {
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gid >= count) return;

	if (ptrs[gid] == nullptr) return;
	delete ptrs[gid];
	ptrs[gid] = nullptr;
}



template <typename T>
template <typename U, typename... Args>
dAbstract<T> dAbstract<T>::MakeAbstract(Args... args) {

	T** ptr2{};
	CUDA_ASSERT(cudaMalloc(&ptr2, sizeof(T*)));

	dAbstract<T> ret(dAbstract<T>::M{ nullptr, ptr2 });

	ret.MakeOnDevice<U>(args...);

	return ret;
}
template <typename T>
dAbstract<T>& dAbstract<T>::operator=(dAbstract<T>&& other) {

	if (this == &other) return *this;

	if (m.ptr2 != nullptr) {
		DeleteOnDevice();
		CUDA_ASSERT(cudaFree(m.ptr2));
	}

	m = std::move(other.m);
	other.m = dAbstract<T>::M::null;

	return *this;
}
template <typename T>
dAbstract<T>::~dAbstract() {
	if (m.ptr2 == nullptr) return;
	DeleteOnDevice();
	CUDA_ASSERT(cudaFree(m.ptr2));
}

template <typename T>
template <typename T2, typename... Args>
void dAbstract<T>::MakeOnDevice(Args... args) {
	_makeOnDevice<T, T2><<<1, 1>>>(m.ptr2, args...);
	CUDA_ASSERT(cudaGetLastError());
	CUDA_ASSERT(cudaMemcpy(&m.ptr, m.ptr2, sizeof(T*), cudaMemcpyDeviceToHost));
}
template <typename T>
void dAbstract<T>::DeleteOnDevice() {
	_deleteOnDevice<<<1, 1>>>(m.ptr2);
	CUDA_ASSERT(cudaDeviceSynchronize());
	m.ptr = nullptr;
}



template <typename T>
dAbstractArray<T> dAbstractArray<T>::MakeArray(size_t count) {

	size_t size = sizeof(T*) * count;
	T** ptr2{};

	CUDA_ASSERT(cudaMalloc(&ptr2, size));
	CUDA_ASSERT(cudaMemset(ptr2, 0, size));

	return dAbstractArray<T>(dAbstractArray<T>::M{ count, ptr2 });
}
template <typename T>
dAbstractArray<T>::~dAbstractArray() {
	if (m.ptr2 != nullptr) {
		DeleteOnDevice(getLength(), 0);
		CUDA_ASSERT(cudaFree(m.ptr2));
	}
}

template <typename T>
dAbstractArray<T>& dAbstractArray<T>::operator=(dAbstractArray<T>&& other) {

	if (this == &other) return *this;

	if (m.ptr2 != nullptr) {
		DeleteOnDevice(getLength(), 0);
		CUDA_ASSERT(cudaFree(m.ptr2));
	}

	m = std::move(other.m);
	other.m = dAbstractArray<T>::M::null;
	
	return *this;
}

template <typename T>
std::vector<T*> dAbstractArray<T>::getPtrVector() const {
	std::vector<T*> vec(length);
	CUDA_ASSERT(cudaMemcpy(vec.data(), ptr2, length * sizeof(T*), cudaMemcpyDeviceToHost));
	return vec;
}

template <typename T>
template <typename T2, typename... Args>
void dAbstractArray<T>::MakeOnDevice(size_t count, size_t array_offset, size_t input_offset, Args*... args) {
	int threads = 32;
	int blocks = ceilDiv(count, threads);
	_makeArrayOnDevice<T, T2, Args...><<<blocks, threads>>>(count, m.ptr2 + array_offset, (args + input_offset)...);
	CUDA_ASSERT(cudaDeviceSynchronize());
}
template <typename T>
template <typename T2, typename... Args>
void dAbstractArray<T>::MakeSingleOnDevice(size_t offset, Args... args) {
	_makeOnDevice<T, T2, Args...><<<1, 1>>>(m.ptr2 + offset, args...);
	CUDA_ASSERT(cudaDeviceSynchronize());
}
template <typename T>
template <typename T2, typename Arg>
void dAbstractArray<T>::MakeOnDeviceVector(size_t count, size_t array_offset, size_t input_offset, const std::vector<Arg>& varg) {
	Arg* d_arg{};
	CUDA_ASSERT(cudaMalloc(&d_arg, sizeof(Arg) * count));
	CUDA_ASSERT(cudaMemcpy(d_arg, varg.data() + input_offset, sizeof(Arg) * count, cudaMemcpyHostToDevice));

	MakeOnDevice<T2, Arg>(count, array_offset, 0, d_arg);

	CUDA_ASSERT(cudaFree(d_arg));
}
template <typename T>
template <typename DeviceFactoryType>
void dAbstractArray<T>::MakeOnDeviceFactoryPtr(size_t count, size_t array_offset, size_t input_offset, DeviceFactoryType* d_factory) {
	int threads = 32;
	int blocks = ceilDiv(count, threads);
	_makeArrayOnDeviceFactory<T, DeviceFactoryType><<<blocks, threads>>>(count, input_offset, m.ptr2 + array_offset, d_factory);
	CUDA_ASSERT(cudaDeviceSynchronize());
}
template <typename T>
template <typename DeviceFactoryType>
void dAbstractArray<T>::MakeOnDeviceFactory(size_t count, size_t array_offset, size_t input_offset, DeviceFactoryType factory) {
	int threads = 32;
	int blocks = ceilDiv(count, threads);
	_makeArrayOnDeviceFactory<T, DeviceFactoryType><<<blocks, threads>>>(count, input_offset, m.ptr2 + array_offset, factory);
	CUDA_ASSERT(cudaDeviceSynchronize());
}
template <typename T>
void dAbstractArray<T>::DeleteOnDevice(size_t count, size_t offset) {
	int threads = 32;
	int blocks = ceilDiv(count, threads);
	_deleteArrayOnDevice<T><<<blocks, threads>>>(count, m.ptr2);
	CUDA_ASSERT(cudaDeviceSynchronize());
}

#endif // HANDLED_DEVICE_ABSTRACT_CLASSES_INL //