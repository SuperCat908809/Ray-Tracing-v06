#ifndef HITTABLE_ABSTRACT_CLASS_H
#define HITTABLE_ABSTRACT_CLASS_H

#include <cuda_runtime.h>
#include "../ray_data.cuh"
#include <concepts>

// abstract class that all hittable classes should inherit from
class Hittable {
protected:
	// protected default constructor, copy constructor and copy assignment,
	//   guaranteeing that no instance of Hittable will ever exist outside of this class
	Hittable() = default;
	Hittable(const Hittable&) = default;
	Hittable& operator=(const Hittable&) = default;

public:
	__device__ virtual bool ClosestIntersection(const Ray& ray, RayPayload& rec) const = 0;
	//__device__ virtual bool OcclusionTest(const Ray& ray) const = 0;
};


class Geometry {
public:

	__host__ __device__ static glm::vec3 getNormal(const Ray& ray, const RayPayload& rec) { return{}; }
	//__host__ __device__ static glm::vec2 getTexCoord(const Ray& ray, const RayPayload& rec) { return {}; }
};

template <typename T>
concept Geometry_t = std::derived_from<T, Geometry>;

#endif // HITTABLE_ABSTRACT_CLASS_H //