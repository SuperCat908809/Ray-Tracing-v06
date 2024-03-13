#ifndef CUDA_RANDOM_GENERATOR_CLASS_H
#define CUDA_RANDOM_GENERATOR_CLASS_H

#include <inttypes.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <glm/glm.hpp>


#if 0
enum GeneratorType{ XORWOW, MRG32k3a, PHILOX_4_32_10, /*MTGP32, */DEFAULT = XORWOW };
template <GeneratorType gen_type, typename ret_type>
class cuRandom {

	template <GeneratorType> struct _genType;
	template <> struct _genType<        XORWOW> { using type =        curandStateXORWOW_t; };
	template <> struct _genType<      MRG32k3a> { using type =      curandStateMRG32k3a_t; };
	template <> struct _genType<PHILOX_4_32_10> { using type = curandStatePhilox4_32_10_t; };
	//template <> struct _genType<        MTGP32> { using type =        curandStateMtgp32_t; };

	using GenType = _genType<gen_type>::type;

	GenType gen;

public:



};
#else

enum DistributionType { UNIFORM, NORMAL, LOG_NORMAL, POISSON, DEFAULT = UNIFORM };
class cuRandom {
	curandStateXORWOW_t gen;

	template <DistributionType D, typename T> __device__ void _RND(T& r1) = 0;
	template <DistributionType D, typename T> __device__ void _RND(T& r1, T& r2) = 0;
	template <DistributionType D, typename T> __device__ void _RND(T& r1, T& r2, T& r3, T& r4) = 0;

	template <DistributionType D, glm::length_t L, typename T, glm::qualifier Q>
	__device__ void _RND4R(glm::vec<L, T, Q>& v) {
		for (int i = 0; i < 4; i += 4) {
			_RND<D, T>(v[i + 0], v[i + 1], v[i + 2], v[i + 3]);
		}
	}

public:

	__device__ cuRandom(uint64_t seed, uint64_t offset = 0, uint64_t sequence = 0) {
		curand_init(seed, sequence, offset, &gen);
	}

	template <typename ret_type> __device__ ret_type nextUniform();
	template <typename ret_type> __device__ ret_type nextNormal();
	template <typename ret_type> __device__ ret_type nextLogNormal(float mean, float stddev);
	template <typename ret_type> __device__ ret_type nextPoisson(float lambda);

	template <DistributionType D, glm::length_t L, typename T, glm::qualifier Q = glm::qualifier::packed_highp>
	__device__ glm::vec<L, T, Q> nextUniformVector() {
		glm::vec<L, T, Q> v;
		if constexpr (L >= 4) {
			_RND4R<D, L, T, Q>(v);
		}
		if constexpr (L & 0b11) {
			const glm::length_t i = L & !0b11;
			if constexpr (i + 1 < L) _RND<D, T>(v[i + 1]);
			if constexpr (i + 2 < L) _RND<D, T>(v[i + 2]);
			if constexpr (i + 3 < L) _RND<D, T>(v[i + 3]);
		}
		return v;
	}

	__device__ void Skipahead(uint64_t n) { skipahead(n, &gen); }
};

template <> __device__ float cuRandom::  nextUniform()							{ return curand_uniform(&gen); }
template <> __device__ float cuRandom::   nextNormal()							{ return curand_normal(&gen); }
template <> __device__ float cuRandom::nextLogNormal(float mean, float stddev)	{ return curand_log_normal(&gen, mean, stddev); }
template <> __device__ float cuRandom::  nextPoisson(float lambda)				{ return curand_poisson(&gen, lambda); }


template <> __device__ void cuRandom::_RND<UNIFORM, float>(float& r1) { r1 = curand_uniform(&gen); }
template <> __device__ void cuRandom::_RND<UNIFORM, float>(float& r1, float& r2) { r1 = curand_uniform(&gen); r2 = curand_uniform(&gen); }
//template <> __device__ void cuRandom::_RND<UNIFORM, float>(float& r1, float& r2, float& r3, float& r4) { float4 R = curand_uniform4(&gen); }
template <> __device__ void cuRandom::_RND<NORMAL, float>(float& r1) { r1 = curand_normal(&gen); }

#endif

#endif // CUDA_RANDOM_GENERATOR_CLASS_H //