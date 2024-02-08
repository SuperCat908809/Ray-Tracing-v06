#ifndef CU_RT_COMMON_H
#define CU_RT_COMMON_H

#include <stdexcept>
#include <assert.h>

#include <glm/glm.hpp>
#include <cuda_runtime.h>


#define _MISS_DIST FLT_MAX

struct Ray {
	glm::vec3 o{ 0,0,0 }, d{ 0,0,1 };

	__host__ __device__ glm::vec3 at(float t) const { return o + d * t; }
};

struct TraceRecord {
	glm::vec3 n{ 0,1,0 };
	float t{ _MISS_DIST };
};

#define CUDA_CHECK(func) cudaAssert(func, #func, __FILE__, __LINE__)
#define CUDA_ASSERT(func) try { CUDA_CHECK(func); } catch (const std::runtime_error& e) { assert(0); }
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

#endif // CU_RT_COMMON_H //