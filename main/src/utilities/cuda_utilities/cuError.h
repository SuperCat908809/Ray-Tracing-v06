#ifndef CUDA_UTILITIES_H
#define CUDA_UTILITIES_H

#include <string>
#include <format>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <stdexcept>

#include <cuda_runtime.h>


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
#define CUDA_ASSERT(func) { cudaError_t _err_code = func; if (cuIsError(_err_code)) { std::cerr << GET_ERR_MSG(func, _err_code) << "\n\n";  assert(_err_code); } }
#endif

#endif // CUDA_UTILITIES_H //