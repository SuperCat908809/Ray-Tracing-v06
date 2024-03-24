#ifndef CUDA_AABB_CLASS_H
#define CUDA_AABB_CLASS_H

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <glm/gtx/component_wise.hpp>
#include "glm_utils.h"
#include "ray_data.cuh"


class aabb {
	glm::vec3 min, max;

public:
	// float maximum in min and float minimum in max guarantees that all rays will 'miss' this aabb and guarantees that 
	// expanding this aabb by another will equal the other, effectively initialising it.
	__host__ __device__ aabb() : min(3.402823466e+38F), max(-3.402823466e+38F) {}
	__host__ __device__ aabb(glm::vec3 min, glm::vec3 max) : min(glm::min(min, max)), max(glm::max(min, max)) {}
	__host__ __device__ aabb(const aabb& a, const aabb& b) : min(glm::min(a.min, b.min)), max(glm::max(a.max, b.max)) {}

	__host__ __device__ glm::vec3 getMin() const { return min; }
	__host__ __device__ glm::vec3 getMax() const { return max; }

	__host__ __device__ aabb& operator+=(const aabb& bounds) { min = glm::min(min, bounds.min); max = glm::max(max, bounds.max); return *this; }

	__host__ __device__ bool intersects(const Ray& ray, float t_max) const {
		glm::vec3 bmin = (min - ray.o) / ray.d;
		glm::vec3 bmax = (max - ray.o) / ray.d;

		glm::vec3 tmp = glm::min(bmin, bmax);
		bmax = glm::max(bmin, bmax);
		bmin = tmp;

		float tmin = glm::compMax(bmin);
		float tmax = glm::compMin(bmax);

		return tmax >= tmin && tmin < t_max && tmax > 0;
	}
};


template <int axis> requires (axis > 0) && (axis <= 3)
__host__ __device__ inline bool box_axis_compare(const aabb& a, const aabb& b) {
	return a.getMin()[axis] < b.getMin()[axis];
}


__host__ __device__ inline bool box_x_compare(const aabb& a, const aabb& b) {
	return a.getMin()[0] < b.getMin()[0];
}

__host__ __device__ inline bool box_y_compare(const aabb& a, const aabb& b) {
	return a.getMin()[1] < b.getMin()[1];
}

__host__ __device__ inline bool box_z_compare(const aabb& a, const aabb& b) {
	return a.getMin()[2] < b.getMin()[2];
}

#endif // CUDA_AABB_CLASS_H //