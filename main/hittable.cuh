#ifndef HITTABLE_ABSTRACT_CLASS_H
#define HITTABLE_ABSTRACT_CLASS_H

#include <cuda_runtime.h>
#include "ray_data.cuh"


// abstract class that all hittable classes should inherit from
class Hittable {

	// delete all constructors, guaranteeing that no instance of Hittable will ever exist directly
	Hittable() = delete;
	Hittable(const Hittable&) = delete;
	Hittable& operator=(const Hittable&) = delete;

public:
	__device__ virtual ~Hittable() {};
	__device__ virtual bool ClosestIntersection(Ray& ray, TraceRecord& rec) const = 0;
};

#endif // HITTABLE_ABSTRACT_CLASS_H //