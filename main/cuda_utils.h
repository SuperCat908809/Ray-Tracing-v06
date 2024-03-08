#ifndef CUDA_UTILITIES_H
#define CUDA_UTILITIES_H

#include <string>
#include <format>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <stdexcept>

#include <cuda_runtime.h>

#if 0
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
#else

inline bool cuIsError(cudaError_t code) { return code != cudaSuccess; }
inline std::string cuFormatErrorMessage(cudaError_t code, const char* func, const char* file, int line) {
	return std::format("GPU assert : {} at {}\n{} {}\n{} :: {}",
		cudaGetErrorName(code), func,
		file, line,
		cudaGetErrorName(code), cudaGetErrorString(code));
}

#define GET_ERR_MSG(func, code) cuFormatErrorMessage(code, #func, __FILE__, __LINE__)
#define CUDA_THROW(func) { cudaError_t _err_code = func; if (cuIsError(_err_code)) { throw std::runtime_error(GET_ERR_MSG(func, _err_code)); } }
#define CUDA_CHECK(func) cuIsError(func)
#ifdef NDEBUG
#define CUDA_ASSERT(func) { func; }
#else
#define CUDA_ASSERT(func) { cudaError_t _err_code = func; if (cuIsError(_err_code)) { std::cerr << GET_ERR_MSG(func, _err_code); assert(0); } }
#endif

#endif

#endif // CUDA_UTILITIES_H //