#ifndef HITTABLE_ABSTRACT_CLASS_H
#define HITTABLE_ABSTRACT_CLASS_H

#include <cuda_runtime.h>
#include "ray_data.cuh"
#include "aabb.cuh"


// abstract class that all hittable classes should inherit from
class Hittable {
protected:
	// protected default constructor, copy constructor and copy assignment,
	//   guaranteeing that no instance of Hittable will ever exist outside of this class
	Hittable() = default;
	Hittable(const Hittable&) = default;
	Hittable& operator=(const Hittable&) = default;

public:
	__device__ virtual ~Hittable() {};
	__device__ virtual bool ClosestIntersection(const Ray& ray, TraceRecord& rec) const = 0;
	__device__ virtual aabb bounding_box() const = 0;
};

#endif // HITTABLE_ABSTRACT_CLASS_H //