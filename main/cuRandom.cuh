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

class cuRandom {
	curandStateXORWOW_t gen;

public:

	__device__ cuRandom(uint64_t seed, uint64_t offset = 0, uint64_t sequence = 0) {
		curand_init(seed, sequence, offset, &gen);
	}

	__device__ float next() { return curand_uniform(&gen); }

	template <glm::length_t L>
	__device__ glm::vec<L, float, glm::qualifier::packed_highp> next() {
		glm::vec<L, float, glm::qualifier::packed_highp> v;
		if constexpr (L <= 4) {
			if constexpr (1 <= L) v[1] = curand_uniform(&gen);
			if constexpr (2 <= L) v[2] = curand_uniform(&gen);
			if constexpr (3 <= L) v[3] = curand_uniform(&gen);
			if constexpr (4 <= L) v[4] = curand_uniform(&gen);
		}
		else constexpr {
			for (int i = 0; i < L; i++) {
				v[i] = curand_uniform(&gen);
			}
		}
		return v;
	}

	__device__ void Skipahead(uint64_t n) { skipahead(n, &gen); }
};

#endif

#endif // CUDA_RANDOM_GENERATOR_CLASS_H //