#ifndef BVH_NODE_CLASS_CUH
#define BVH_NODE_CLASS_CUH

#include <cuda_runtime.h>
#include "hittable.cuh"
#include "aabb.cuh"


class bvh_node : public Hittable {
	aabb bounds;
	Hittable* left;
	Hittable* right;

public:

	__device__ bvh_node() : left(nullptr), right(nullptr), bounds() {}
	__device__ bvh_node(Hittable* left, Hittable* right, const aabb& bounds) : left(left), right(right), bounds(bounds) {}

	__device__ virtual bool ClosestIntersection(const Ray& ray, RayPayload& rec) const override {
		if (!bounds.intersects(ray, rec.distance)) return false;
		bool hit = left->ClosestIntersection(ray, rec);
		hit |= right->ClosestIntersection(ray, rec);
		return hit;
	}
};

#endif // BVH_NODE_CLASS_CUH //