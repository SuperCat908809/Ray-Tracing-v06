#ifndef HANDLED_DEVICE_ABSTRACT_CLASSES_H
#define HANDLED_DEVICE_ABSTRACT_CLASSES_H

#include "cuda_utils.h"

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

template <typename T>
__global__ inline void _deleteArrayOnDevice(size_t count, T** ptrs) {
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gid >= count) return;

	if (ptrs[gid] == nullptr) return;
	delete ptrs[gid];
	ptrs[gid] = nullptr;
}


template <typename T>
class HandledDeviceAbstract {
	T* ptr{};
	T** ptr2{};

public:

	HandledDeviceAbstract(const HandledDeviceAbstract&) = delete;
	HandledDeviceAbstract& operator=(const HandledDeviceAbstract&) = delete;

	HandledDeviceAbstract() {
		CUDA_ASSERT(cudaMalloc(&ptr2, sizeof(T*)));
		CUDA_ASSERT(cudaMemset(ptr2, 0, sizeof(T*)));
		ptr = nullptr;
	}
	template <typename... Args>
	HandledDeviceAbstract(Args... args) {
		CUDA_ASSERT(cudaMalloc(&ptr2, sizeof(T*)));
		MakeOnDevice<T>(args...);
	}
	~HandledDeviceAbstract() {
		DeleteOnDevice();
		CUDA_ASSERT(cudaFree(ptr2));
	}

	template <typename T2, typename... Args>
	void MakeOnDevice(Args... args) {
		_makeOnDevice<T, T2><<<1, 1>>>(ptr2, args...);
		CUDA_ASSERT(cudaGetLastError());
		CUDA_ASSERT(cudaMemcpy(&ptr, ptr2, sizeof(T*), cudaMemcpyDeviceToHost));
	}
	void DeleteOnDevice() {
		_deleteOnDevice<<<1, 1>>>(ptr2);
		CUDA_ASSERT(cudaDeviceSynchronize());
		ptr = nullptr;
	}

	T* getPtr() const { return ptr; }
	T** getPtr2() const { return ptr2; }
};


template <typename T>
class HandledDeviceAbstractArray {
	std::vector<T*> ptrs{};
	T** ptr2{};

public:

	HandledDeviceAbstractArray() = delete;
	HandledDeviceAbstractArray(const HandledDeviceAbstractArray&) = delete;
	HandledDeviceAbstractArray& operator=(const HandledDeviceAbstractArray&) = delete;

	HandledDeviceAbstractArray(size_t count) {
		size_t size = sizeof(T*) * count;
		CUDA_ASSERT(cudaMalloc(&ptr2, size));
		CUDA_ASSERT(cudaMemset(ptr2, 0, size));
		ptrs.resize(count);
		memset(ptrs.data(), 0, size);
	}
	~HandledDeviceAbstractArray() {
		DeleteOnDevice(getSize(), 0);
		CUDA_ASSERT(cudaFree(ptr2));
	}

	size_t getSize() const { return ptrs.size(); }
	T** getDeviceArrayPtr() const { return ptr2; }
	std::vector<T*> getPtrVector() const { return ptrs; }

	template <typename T2, typename... Args>
	void MakeOnDevice(size_t count, size_t array_offset, size_t input_offset, Args*... args) {
		int threads = 32;
		int blocks = ceilDiv(count, threads);
		_makeArrayOnDevice<T, T2, Args...><<<blocks, threads>>>(count, ptr2 + array_offset, (args + input_offset)...);
		CUDA_ASSERT(cudaDeviceSynchronize());
		CUDA_ASSERT(cudaMemcpy(ptrs.data() + array_offset, ptr2 + array_offset, sizeof(T*) * count, cudaMemcpyDeviceToHost));
	}
	template <typename T2, typename Arg>
	void MakeOnDeviceVector(size_t count, size_t array_offset, size_t input_offset, std::vector<Arg> varg) {
		Arg* d_arg{};
		CUDA_ASSERT(cudaMalloc(&d_arg, sizeof(Arg) * count));
		CUDA_ASSERT(cudaMemcpy(d_arg, varg.data() + input_offset, sizeof(Arg) * count, cudaMemcpyHostToDevice));

		MakeOnDevice<T2, Arg>(count, array_offset, 0, d_arg);

		CUDA_ASSERT(cudaFree(d_arg));
	}
	template <typename DeviceFactoryType>
	void MakeOnDeviceFactory(size_t count, size_t array_offset, size_t input_offset, DeviceFactoryType* d_factory) {
		int threads = 32;
		int blocks = ceilDiv(count, threads);
		_makeArrayOnDeviceFactory<T, DeviceFactoryType><<<blocks, threads>>>(count, input_offset, ptr2 + array_offset, d_factory);
		CUDA_ASSERT(cudaDeviceSynchronize());
		CUDA_ASSERT(cudaMemcpy(ptrs.data() + array_offset, ptr2 + array_offset, sizeof(T*) * count, cudaMemcpyDeviceToHost));
	}
	void DeleteOnDevice(size_t count, size_t offset) {
		int threads = 32;
		int blocks = ceilDiv(count, threads);
		_deleteArrayOnDevice<T><<<blocks, threads>>>(count, ptr2);
		CUDA_ASSERT(cudaDeviceSynchronize());
		memset(ptrs.data() + offset, 0, sizeof(T*) * count);
	}
};

#endif // HANDLED_DEVICE_ABSTRACT_CLASSES_H //