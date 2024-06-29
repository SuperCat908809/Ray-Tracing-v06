// pch.h: This is a precompiled header file.
// Files listed below are compiled only once, improving build performance for future builds.
// This also affects IntelliSense performance, including code completion and many code browsing features.
// However, files listed here are ALL re-compiled if any one of them is updated between builds.
// Do not add files here that you will be updating frequently as this negates the performance advantage.

#ifndef PCH_H
#define PCH_H

// add headers that you want to pre-compile here

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "utilities/glm_utils.h"
#include "utilities/timers.h"
#include "utilities/cuda_utilities/cuError.h"
#include "utilities/cuda_utilities/cuHostRND.h"

#endif //PCH_H
