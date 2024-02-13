#ifndef CU_RT_COMMON_H
#define CU_RT_COMMON_H

#define _USE_MATH_DEFINES
#include <math.h>
#include <limits>
#include <memory>
#include <assert.h>
#include <stdexcept>

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define _MISS_DIST FLT_MAX

#include "ray_data.cuh"
#include "cuda_utils.h"
#include "glm_utils.h"

inline constexpr int ceilDiv(int n, int d) { return (n + d - 1) / d; }

#define RND (curand_uniform(random_state))
#define RNDR(min, max) (curand_uniform(random_state) * (max - min) + min)
#define RND3 (glm::cu_random_uniform<3>(random_state))
#define RNDR3(min, max) (glm::map(RND3, min, max))
#define RND_IN_SPHERE (glm::cu_random_in_unit_vec<3>(random_state))
#define RND_ON_SPHERE (glm::cu_random_unit_vec<3>(random_state))

#endif // CU_RT_COMMON_H //