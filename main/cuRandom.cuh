#ifndef CUDA_RANDOM_GENERATOR_CLASS_H
#define CUDA_RANDOM_GENERATOR_CLASS_H

#include <inttypes.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>


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

class cuRandom {
	curandStateXORWOW_t gen;

public:

	__device__ cuRandom(uint64_t seed, uint64_t offset = 0, uint64_t sequence = 0) {
		curand_init(seed, sequence, offset, &gen);
	}

	template <typename ret_type> __device__ ret_type nextUniform();
	template <typename ret_type> __device__ ret_type nextNormal();
	template <typename ret_type> __device__ ret_type nextLogNormal(float mean, float stddev);
	template <typename ret_type> __device__ ret_type nextPoisson(float lambda);

	__device__ void Skipahead(uint64_t n) { skipahead(n, &gen); }
};

template <> __device__ float cuRandom::  nextUniform()							{ return curand_uniform(&gen); }
template <> __device__ float cuRandom::   nextNormal()							{ return curand_normal(&gen); }
template <> __device__ float cuRandom::nextLogNormal(float mean, float stddev)	{ return curand_log_normal(&gen, mean, stddev); }
template <> __device__ float cuRandom::  nextPoisson(float lambda)				{ return curand_poisson(&gen, lambda); }

#endif

#endif // CUDA_RANDOM_GENERATOR_CLASS_H //