#ifndef HITTABLE_LIST_CLASS_H
#define HITTABLE_LIST_CLASS_H

#include <cuda_runtime.h>
#include "ray_data.cuh"
#include "hittable.cuh"
#include "aabb.cuh"


class HittableList : public Hittable {
	Hittable** objects{};
	int object_count{};
	aabb bounds;

	__device__ HittableList() = default;
	__device__ HittableList(const HittableList&) = default;
public:

	__device__ HittableList(Hittable** objects, int object_count, const aabb& bounds) : objects(objects), object_count(object_count), bounds(bounds) {}

	__device__ virtual bool ClosestIntersection(const Ray& ray, TraceRecord& rec) const override {
		if (!bounds.intersects(ray, rec.t)) return false;

		bool hit_any{ false };

		for (int i = 0; i < object_count; i++) {
			hit_any |= objects[i]->ClosestIntersection(ray, rec);
		}
		// rec only gets updated when an intersection has been found.

		return hit_any;
	}
};

#endif // HITTABLE_LIST_CLASS_H //