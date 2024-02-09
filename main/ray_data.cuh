#ifndef RAY_DATA_STRUCTS_H
#define RAY_DATA_STRUCTS_H

#include <glm/glm.hpp>
#include <cuda_runtime.h>

struct Ray {
	glm::vec3 o{ 0,0,0 }, d{ 0,0,1 };

	__host__ __device__ glm::vec3 at(float t) const { return o + d * t; }
};

struct TraceRecord {
	glm::vec3 n{ 0,1,0 };
	float t{ _MISS_DIST };
	bool hit_backface{ false };
};

#endif // RAY_STRUCT_H //