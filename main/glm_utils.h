#ifndef GLM_UTILS_H
#define GLM_UTILS_H

#include <glm/glm.hpp>

namespace glm {
	
	template <glm::length_t L, typename T, glm::qualifier Q>
	inline constexpr bool near_zero(const glm::vec<L, T, Q>& v, T epsilon = 1e-6f) {
		auto v2 = glm::abs(v);
		for (int i = 0; i < L; i++) {
			if (v2[i] > epsilon) {
				return false;
			}
		}
		return true;
	}

	template <glm::length_t L, typename T, glm::qualifier Q>
	inline constexpr T length2(const glm::vec<L, T, Q>& v) {
		T sum{};
		for (int i = 0; i < L; i++) {
			sum += v[i] * v[i];
		}
		return sum;
	}

	template <glm::length_t L, typename T, glm::qualifier Q>
	inline constexpr T distance2(const glm::vec<L, T, Q>& v1, const glm::vec<L, T, Q>& v2) {
		return length2(v2 - v1);
	}

};

#endif // GLM_UTILS_H //