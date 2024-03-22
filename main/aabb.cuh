#ifndef CUDA_AABB_CLASS_H
#define CUDA_AABB_CLASS_H

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include "glm_utils.h"
#include "ray_data.cuh"


class aabb {
	glm::vec3 min, max;

public:
	__host__ __device__ aabb() : min(0), max(0) {}
	__host__ __device__ aabb(glm::vec3 min, glm::vec3 max) : min(glm::min(min, max)), max(glm::max(min, max)) {}
	__host__ __device__ aabb(const aabb& a, const aabb& b) : min(glm::min(a.min, b.min)), max(glm::max(a.max, b.max)) {}

	__host__ __device__ bool intersects(const Ray& ray, float t_max) const {
		glm::vec3 bmin = (min - ray.o) / ray.d;
		glm::vec3 bmax = (max - ray.o) / ray.d;

		glm::vec3 tmp = glm::min(bmin, bmax);
		bmax = glm::max(bmin, bmax);
		bmin = tmp;

		float tmin = glm::compwise_max(bmin);
		float tmax = glm::compwise_min(bmax);

		return tmax >= tmin && tmin < t_max && tmax > 0;
	}
};

#endif // CUDA_AABB_CLASS_H //