#include <tuple>
#include <vector>
#include <utility>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <stdexcept>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <glm/glm.hpp>


#define CUDA_CHECK(func) cudaAssert(func, #func, __FILE__, __LINE__)
#define CUDA_ASSERT(func) try { CUDA_CHECK(func); } catch (const std::runtime_error&) { assert(0); }
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

constexpr int ceilDiv(int numer, int denom) { return (numer + denom - 1) / denom; }

template <typename T, typename... Args>
__global__ inline void _makeOnDevice(T** ptr, Args... args) {
	if (!(threadIdx.x == 0 && blockIdx.x == 0)) return;

	(*ptr) = new T(args...);
}

template <typename T>
__global__ inline void _deleteOnDevice(T** ptr) {
	if (!(threadIdx.x == 0 && blockIdx.x == 0)) return;

	if (*ptr == nullptr) return;
	delete* ptr;
	ptr = nullptr;
}


template <typename T>
class dAbstract {
	T* ptr{};
	T** ptr2{};

public:

	dAbstract() = delete;
	dAbstract(const dAbstract&) = delete;
	dAbstract& operator=(const dAbstract&) = delete;

	template <typename... Args>
	dAbstract(Args... args) {
		CUDA_ASSERT(cudaMalloc(&ptr2, sizeof(T*)));
		_makeOnDevice<<<1, 1>>>(ptr2, args...);
		CUDA_ASSERT(cudaGetLastError());
		CUDA_ASSERT(cudaMemcpy(&ptr, ptr2, sizeof(T*), cudaMemcpyDeviceToHost));
	}
	__host__ ~dAbstract() {
		_deleteOnDevice<<<1, 1>>>(ptr2);
		CUDA_ASSERT(cudaDeviceSynchronize());
		CUDA_ASSERT(cudaFree(ptr2));
	}

	T* getPtr() const { return ptr; }
};



template <typename T1, typename T2, typename... Args>
__global__ inline void _makeArrayOnDevice(int count, T1** ptrs, Args*... args) {
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gid >= count) return;

	ptrs[gid] = new T2(args[gid]...);
}

template <typename T>
__global__ inline void _deleteArrayOnDevice(int count, T** ptrs) {
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gid >= count) return;

	if (ptrs[gid] == nullptr) return;
	delete ptrs[gid];
	ptrs[gid] = nullptr;
}

template <typename T>
class HandledDeviceAbstractArray {
	std::vector<T*> ptrs{};
	T** ptr2{};

public:

	HandledDeviceAbstractArray(size_t count) {
		size_t size = sizeof(T*) * count;
		CUDA_ASSERT(cudaMalloc(&ptr2, size));
		CUDA_ASSERT(cudaMemset(ptr2, 0, size));
		ptrs.resize(count);
		memset(ptrs.data(), 0, size);
	}
	~HandledDeviceAbstractArray() {
		int count = (int)getLength();
		int threads = 32;
		int blocks = ceilDiv(count, threads);
		_deleteArrayOnDevice<T><<<blocks, threads>>>(count, ptr2);
		CUDA_ASSERT(cudaDeviceSynchronize());
		CUDA_ASSERT(cudaFree(ptr2));
	}

	size_t getLength() const { return ptrs.size(); }
	T** getDevicePtrArray() const { return ptr2; }
	std::vector<T*> getPtrVector() const { return ptrs; }

	template <typename T2, typename... Args>
	void MakeOnDevice(size_t count, size_t offset, Args*... args) {
		int threads = 32;
		int blocks = ceilDiv(count, threads);
		_makeArrayOnDevice<T, T2, Args...><<<blocks, threads>>>(count, ptr2 + offset, args...);
		CUDA_ASSERT(cudaDeviceSynchronize());
		CUDA_ASSERT(cudaMemcpy(ptrs.data() + offset, ptr2 + offset, sizeof(T*) * count, cudaMemcpyDeviceToHost));
	}
	template <typename T2, typename Arg>
	void MakeOnDeviceVector(size_t count, size_t offset, std::vector<Arg> varg) {

		Arg* d_arg{};
		CUDA_ASSERT(cudaMalloc(&d_arg, sizeof(Arg) * count));
		CUDA_ASSERT(cudaMemcpy(d_arg, varg.data(), sizeof(Arg) * count, cudaMemcpyHostToDevice));

		int threads = 32;
		int blocks = ceilDiv(count, threads);
		_makeArrayOnDevice<T, T2, Arg><<<blocks, threads>>>(count, ptr2 + offset, d_arg);
		CUDA_ASSERT(cudaDeviceSynchronize());
		CUDA_ASSERT(cudaMemcpy(ptrs.data() + offset, ptr2 + offset, sizeof(T*) * count, cudaMemcpyDeviceToHost));

		CUDA_ASSERT(cudaFree(d_arg));
	}
	void DeleteOnDevice(size_t count, size_t offset = 0) {
		int threads = 32;
		int blocks = ceilDiv(count, threads);
		_deleteArrayOnDevice<T><<<blocks, threads>>>(count, ptr2);
		CUDA_ASSERT(cudaDeviceSynchronize());
		memset(ptrs.data() + offset, 0, sizeof(T*) * count);
	}
};



class Base {
public:
	__device__ virtual ~Base() {};
	__device__ virtual void func() const = 0;
};

class Child1 : public Base {
	int i1{}, i2{};
public:
	struct params {
		int i1{};
		int i2{};
	};

	__device__ Child1(const params& p) : i1(p.i1), i2(p.i2) {
		printf("Constructing Child 1\n");
	}
	__device__ Child1(int i1, int i2) : i1(i1), i2(i2) {
		printf("Constructing Child 1\n");
	}
	__device__ virtual ~Child1() override {
		printf("Deconstructing Child 1\n");
	}
	__device__ virtual void func() const override {
		printf("Child 1.func %i, %i\n", i1, i2);
	}
};

class Child2 : public Base {
	int i{};
public:
	struct params {
		int i{};
	};

	__device__ Child2(const params& p) : i(p.i) {
		printf("Constructing Child 2\n");
	}
	__device__ Child2(int i) : i(i) {
		printf("Constructing Child 2\n");
	}
	__device__ virtual ~Child2() override {
		printf("Deconstructing Child 2\n");
	}
	__device__ virtual void func() const override {
		printf("Child 2.func %i\n", i);
	}
};

__global__ inline void BaseFunc(Base* ptr) {
	if (!(threadIdx.x == 0 && blockIdx.x == 0)) return;

	ptr->func();
}

template <typename T>
__global__ inline void BaseFuncv(int count, T** ptr) {
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gid >= count) return;

	if (ptr[gid] == nullptr) return;
	ptr[gid]->func();
}

int main() {


#if 0
	auto c1i1v = std::vector<int>{ 0, 1, 2, 3, 4 };
	auto c1i2v = std::vector<int>{ 8, 9, 10, 11, 12 };

	auto c2iv = std::vector<int>{ -4, -5, -6, -7, -8 };

	int* d_c1i1v{};
	int* d_c1i2v{};
	CUDA_ASSERT(cudaMalloc(&d_c1i1v, sizeof(int) * 5));
	CUDA_ASSERT(cudaMalloc(&d_c1i2v, sizeof(int) * 5));
	CUDA_ASSERT(cudaMemcpy(d_c1i1v, c1i1v.data(), sizeof(int) * 5, cudaMemcpyHostToDevice));
	CUDA_ASSERT(cudaMemcpy(d_c1i2v, c1i2v.data(), sizeof(int) * 5, cudaMemcpyHostToDevice));

	int* d_c2iv{};
	CUDA_ASSERT(cudaMalloc(&d_c2iv, sizeof(int) * 5));
	CUDA_ASSERT(cudaMemcpy(d_c2iv, c2iv.data(), sizeof(int) * 5, cudaMemcpyHostToDevice));

	
	auto basev = new HandledDeviceAbstractArray<Base>(10);
	basev->MakeOnDevice<Child1>(5, 0, d_c1i1v, d_c1i2v);
	basev->MakeOnDevice<Child2>(5, 5, d_c2iv);

	CUDA_ASSERT(cudaFree(d_c1i1v));
	CUDA_ASSERT(cudaFree(d_c1i2v));
	CUDA_ASSERT(cudaFree(d_c2iv));


	BaseFuncv<Base><<<1, 10>>>(10, basev->getDevicePtrArray());
	CUDA_ASSERT(cudaDeviceSynchronize());

	delete basev;
#else
	auto c1v = std::vector<Child1::params>{
		{0, 8}, {1,9},{2,10},{3,11},{4,12},
	};
	auto c2v = std::vector<Child2::params>{
		{-4},{-5},{-6},{-7},{-8}
	};

	auto basev = new HandledDeviceAbstractArray<Base>(10);
	basev->MakeOnDeviceVector<Child1>(5, 0, c1v);
	basev->MakeOnDeviceVector<Child2>(5, 5, c2v);

	BaseFuncv<Base><<<1, 10>>>(10, basev->getDevicePtrArray());
	CUDA_ASSERT(cudaDeviceSynchronize());

	delete basev;
#endif

	CUDA_ASSERT(cudaDeviceReset());

	std::cout << "\n\nFinished\n.";
	return 0;
}