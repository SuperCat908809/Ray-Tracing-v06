#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_utils.h"

struct base {

	__device__ virtual void func() const = 0;
};

struct derived1 : public base {

	int i;

	__device__ derived1(int i) : i(i) {}

	__device__ virtual void func() const override {
		printf("derived1.func, i %i\n", i);
	}
};

struct derived2 : public base {

	float f;

	__device__ derived2(float f) : f(f) {}

	__device__ virtual void func() const override {
		printf("derived2.func, f %f\n", f);
	}
};

__global__ void kernel(base* d1, base* d2) {
	if (!(threadIdx.x == 0 && blockIdx.x == 0)) return;

	(d1)->func();
	(d2)->func();
}


int main() {

	HandledDeviceAbstract<derived1>* d1 = new HandledDeviceAbstract<derived1>(2   );
	HandledDeviceAbstract<derived2>* d2 = new HandledDeviceAbstract<derived2>(3.0f);

	kernel<<<1, 1>>>(d1->getPtr(), d2->getPtr());
	CUDA_ASSERT(cudaDeviceSynchronize());
	
	delete d1;
	delete d2;


	CUDA_ASSERT(cudaDeviceReset());

	std::cout << "\n\nFinished\n.";
	return 0;
}