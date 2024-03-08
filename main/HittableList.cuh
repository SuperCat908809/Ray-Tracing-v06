#ifndef HITTABLE_LIST_CLASS_H
#define HITTABLE_LIST_CLASS_H

#include <cuda_runtime.h>
#include "ray_data.cuh"
#include "hittable.cuh"


class HittableList : public Hittable {
	Hittable** objects{};
	int object_count{};

public:
	__device__ HittableList() = delete;
	__device__ HittableList(const HittableList&) = delete;
	__device__ HittableList& operator=(const HittableList&) = delete;

	__device__ HittableList(Hittable** objects, int object_count) : objects(objects), object_count(object_count) {}

	__device__ virtual bool ClosestIntersection(Ray& ray, TraceRecord& rec) const override {
		bool hit_any{ false };

		for (int i = 0; i < object_count; i++) {
			hit_any |= objects[i]->ClosestIntersection(ray, rec);
		}
		// rec only gets updated when an intersection has been found.
		// we want to discard the last rec if a closer one is found.
		// hence passing rec itself is ok since a further intersection would be disgarded for a closer one.

		return hit_any;
	}
};

#endif // HITTABLE_LIST_CLASS_H //