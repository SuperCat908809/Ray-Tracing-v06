#ifndef HITTABLE_ABSTRACT_CLASS_H
#define HITTABLE_ABSTRACT_CLASS_H

#include <cuda_runtime.h>
#include "ray_data.cuh"


// abstract class that all hittable classes should inherit from
class Hittable {
public:
	__device__ virtual ~Hittable() {};
	__device__ virtual bool ClosestIntersection(Ray& ray, TraceRecord& rec) const = 0;
};

#endif // HITTABLE_ABSTRACT_CLASS_H //