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

#include "ray_data.cuh"
#include "../utilities/cuda_utilities/cuError.h"
#include "../utilities/glm_utils.h"

inline constexpr int ceilDiv(int n, int d) { return (n + d - 1) / d; }

#endif // CU_RT_COMMON_H //