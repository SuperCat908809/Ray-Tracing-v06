#ifndef CUDA_AABB_CLASS_H
#define CUDA_AABB_CLASS_H

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <glm/gtx/component_wise.hpp>
#include "../../utilities/glm_utils.h"
#include "../ray_data.cuh"


class aabb {
	glm::vec3 min, max;

public:
	// float maximum in min and float minimum in max guarantees that all rays will 'miss' this aabb and guarantees that 
	// expanding this aabb by another will equal the other, effectively initialising it.
	__host__ __device__ aabb() : min(1e9f), max(-1e9f) {}
	__host__ __device__ aabb(glm::vec3 min, glm::vec3 max) : min(min), max(max) {}
	__host__ __device__ aabb(const aabb& a, const aabb& b) : min(glm::min(a.min, b.min)), max(glm::max(a.max, b.max)) {}

	__host__ __device__ glm::vec3 getMin() const { return min; }
	__host__ __device__ glm::vec3 getMax() const { return max; }

	__host__ __device__ aabb& operator+=(const aabb& bounds) { min = glm::min(min, bounds.min); max = glm::max(max, bounds.max); return *this; }

	__host__ __device__ bool intersects(const Ray& ray, float ray_max_dist) const {
		float dist{};
		return intersects(ray, ray_max_dist, dist);
	}
	__host__ __device__ bool intersects(const Ray& ray, float ray_max_dist, float& dist) const {
		glm::vec3 bmin = (min - ray.o) / ray.d;
		glm::vec3 bmax = (max - ray.o) / ray.d;

		glm::vec3 tmp_min = glm::min(bmin, bmax);
		bmax = glm::max(bmin, bmax);
		bmin = tmp_min;

		float tmin = glm::compMax(bmin);
		float tmax = glm::compMin(bmax);

		bool hit = tmin <= tmax && tmin < ray_max_dist && tmax > 0;
		if (hit) dist = tmin;
		return hit;
	}

	__host__ __device__ int longest_axis() const {
		glm::vec3 span = glm::abs(max - min);

		if (span.x > span.y)
			return span.x > span.z ? 0 : 2;
		else
			return span.y > span.z ? 1 : 2;
	}

	__host__ __device__ float surface_area() const {
		glm::vec3 span = max - min;
		if (span.x < 0 || span.y < 0 || span.z < 0) return 0.0f;

		float cost = 0.0f;
		cost += span.x * span.y;
		cost += span.x * span.z;
		cost += span.y * span.z;
		return 2.0f * cost;
	}

	__host__ __device__ glm::vec3 centeroid() const {
		return (max + min) * 0.5f;
	}
};


template <int axis> requires (axis >= 0) && (axis < 3)
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