#ifndef RAY_DATA_STRUCTS_H
#define RAY_DATA_STRUCTS_H

#include <glm/glm.hpp>
#include <cuda_runtime.h>

struct Ray {
	glm::vec3 o{ 0,0,0 }, d{ 0,0,1 };

	__host__ __device__ Ray() = default;
	__host__ __device__ Ray(glm::vec3 origin, glm::vec3 direction) : o(origin), d(direction) {}
	__host__ __device__ glm::vec3 at(float t) const { return o + d * t; }
};

class Material;
struct TraceRecord {
	glm::vec3 n{ 0,1,0 };
	float t{ _MISS_DIST };
	Material* mat_ptr{ nullptr };
	bool hit_backface{ false };

	__host__ __device__ void set_face_normal(const Ray& r, const glm::vec3& outward_normal) {
		hit_backface = glm::dot(r.d, outward_normal) > 0;
		n = hit_backface ? -outward_normal : outward_normal;
	}
};

#endif // RAY_STRUCT_H //