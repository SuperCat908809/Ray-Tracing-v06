#ifndef GLM_UTILS_H
#define GLM_UTILS_H

#include <glm/glm.hpp>
#include <concepts>

#ifdef __CUDA_RUNTIME_H__
#define GLM_UTIL_CUDA_BOTH __host__ __device__
#else
#define GLM_UTIL_CUDA_BOTH
#endif

namespace glm {

	template <glm::length_t L, typename T, glm::qualifier Q>
	GLM_UTIL_CUDA_BOTH inline constexpr bool near_zero(const glm::vec<L, T, Q>& v, T epsilon = 1e-9f) {
		auto v2 = glm::abs(v);
#pragma unroll
		for (int i = 0; i < L; i++) {
			if (v2[i] > epsilon) {
				return false;
			}
		}
		return true;
	}

	template <glm::length_t L, typename T, glm::qualifier Q>
	GLM_UTIL_CUDA_BOTH inline constexpr T length2(const glm::vec<L, T, Q>& v) {
		T sum{};
#pragma unroll
		for (int i = 0; i < L; i++) {
			sum += v[i] * v[i];
		}
		return sum;
	}

	template <glm::length_t L, typename T, glm::qualifier Q>
	GLM_UTIL_CUDA_BOTH inline constexpr T distance2(const glm::vec<L, T, Q>& v1, const glm::vec<L, T, Q>& v2) {
		return length2(v2 - v1);
	}

	template <glm::length_t L, typename T, glm::qualifier Q>
	GLM_UTIL_CUDA_BOTH inline constexpr glm::vec<L, T, Q> map(const glm::vec<L, T, Q>& val, const glm::vec<L, T, Q>& min, const glm::vec<L, T, Q>& max) {
		return val * (max - min) + min;
	}
	template <glm::length_t L, typename T, glm::qualifier Q>
	GLM_UTIL_CUDA_BOTH inline constexpr glm::vec<L, T, Q> invmap(const glm::vec<L, T, Q>& val, const glm::vec<L, T, Q>& min, const glm::vec<L, T, Q>& max) {
		return (val - min) / (max - min);
	}

	template <glm::length_t L, typename T, glm::qualifier Q>
	GLM_UTIL_CUDA_BOTH inline constexpr glm::vec<L, T, Q> map(const glm::vec<L, T, Q>& val,
		const glm::vec<L, T, Q>& src_min, const glm::vec<L, T, Q>& src_max,
		const glm::vec<L, T, Q>& dst_min, const glm::vec<L, T, Q>& dst_max) {
		return map(invmap(val, src_min, src_max), dst_min, dst_max);
	}
	template <glm::length_t L, typename T, glm::qualifier Q>
	GLM_UTIL_CUDA_BOTH inline constexpr glm::vec<L, T, Q> invmap(const glm::vec<L, T, Q>& val,
		const glm::vec<L, T, Q>& src_min, const glm::vec<L, T, Q>& src_max,
		const glm::vec<L, T, Q>& dst_min, const glm::vec<L, T, Q>& dst_max) {
		return map(val, dst_min, dst_max, src_min, src_max);
	}

	template <glm::length_t L, typename T, glm::qualifier Q>
	GLM_UTIL_CUDA_BOTH inline constexpr glm::vec<L, T, Q> linear_interpolate(const glm::vec<L, T, Q>& a, const glm::vec<L, T, Q>& b, const T& factor) {
		return a + (b - a) * factor;
	}

};

#endif // GLM_UTILS_H //


#if !defined(GLM_CURAND_UTILS_H) && defined(CURAND_KERNEL_H_)
#define GLM_CURAND_UTILS_H

#include <curand_kernel.h>
#include <glm/glm.hpp>
#include "cuda_utilities/cuRandom.cuh"


namespace glm {

	template <glm::length_t L, glm::qualifier Q = glm::packed_highp>
	__device__ inline glm::vec<L, float, Q> cuRandomInUnit(cuRandom& rnd) {
		do {
			glm::vec<L, float, Q> rnd_v = rnd.next<L>() * 2.0f - 1.0f;
			if (glm::length2(rnd_v) < 1.0f) return rnd_v;
		} while (true);
	}

	template <glm::length_t L, glm::qualifier Q = glm::packed_highp>
	__device__ inline glm::vec<L, float, Q> cuRandomOnUnit(cuRandom& rnd) {
		do {
			glm::vec<L, float, Q> rnd_v = rnd.next<L>() * 2.0f - 1.0f;
			if (!glm::near_zero(rnd_v) && glm::length2(rnd_v) < 1.0f) return glm::normalize(rnd_v);
		} while (true);
	}

};

#endif // GLM_CURAND_UTILS_H //
