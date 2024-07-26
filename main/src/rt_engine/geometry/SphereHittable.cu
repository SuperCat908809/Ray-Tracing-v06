#include "SphereHittable.cuh"

#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "../../utilities/cuda_utilities/cuda_utils.cuh"

#include "../ray_data.cuh"
#include "aabb.cuh"
#include "hittable.cuh"
#include "../shaders/material.cuh"


#if 0
__host__ __device__ float _sphere_closest_intersection(const Ray& ray, glm::vec3 center, float radius) {
	glm::vec3 oc = ray.o - center;

	float a = glm::dot(ray.d, ray.d);
	float hb = glm::dot(ray.d, oc);
	float c = glm::dot(oc, oc) - radius * radius;
	float d = hb * hb - a * c;
	if (d <= 0) return _MISS_DIST;

	d = sqrtf(d);
	float t = (-hb - d) / a;
	if (t < 0.0f) {
		t = (-hb + d) / a;
		if (t < 0.0f)
			return _MISS_DIST;
	}

	return t;
}
#endif



static_assert(sizeof(Sphere::TraceRecord) <= sizeof(RayPayload::Payload),
	"Ray payload is too small to fit SphereHittable::TraceRecord");
static_assert(alignof(Sphere::TraceRecord) <= alignof(RayPayload::Payload),
	"Ray payload alignment is too small to fit SphereHittable::TraceRecord");

__host__ __device__ glm::vec3 Sphere::getNormal(const Ray& ray, const RayPayload& rec) {
	auto& sp_rec = *reinterpret_cast<const Sphere::TraceRecord*>(&rec.payload);
	return sp_rec.normal;
	//const Sphere& sp = *sp_rec.sphere;
	//glm::vec3 intersect_pos = ray.at(rec.distance);
	//glm::vec3 normal = (intersect_pos - sp.center) / sp.radius;
	//return normal;
}

__host__ __device__  aabb getSphereBounds(const Sphere& sp) {
	return aabb(sp.center - glm::vec3(sp.radius), sp.center + glm::vec3(sp.radius));
}

__device__ bool SphereHittable::ClosestIntersection(const Ray& ray, RayPayload& rec) const {
	float t = _sphere_closest_intersection(ray, sphere->center, sphere->radius);
	if (t >= rec.distance) return false;

	rec.material_ptr = mat_ptr;
	rec.distance = t;
	auto& sp_rec = *reinterpret_cast<Sphere::TraceRecord*>(&rec.payload);
	sp_rec.sphere = sphere;
	sp_rec.normal = (ray.at(rec.distance) - sphere->center) / sphere->radius;
	return true;
}



static_assert(sizeof(MovingSphere::TraceRecord) <= sizeof(RayPayload::Payload),
	"Ray payload is too small to fit MovingSphereHittable::TraceRecord");
static_assert(alignof(MovingSphere::TraceRecord) <= alignof(RayPayload::Payload),
	"Ray payload alignment is too small to fit MovingSphereHittable::TraceRecord");

__host__ __device__ glm::vec3 MovingSphere::getNormal(const Ray& ray, const RayPayload& rec) {
	auto& sp_rec = *reinterpret_cast<const MovingSphere::TraceRecord*>(&rec.payload);
	return sp_rec.normal;
	//const MovingSphere& sp = *sp_rec.moving_sphere;
	//glm::vec3 intersect_pos = ray.at(rec.distance);
	//glm::vec3 time_sliced_center = glm::mix(sp.center0, sp.center1, ray.time);
	//glm::vec3 normal = (intersect_pos - time_sliced_center) / sp.radius;
	//return normal;
}

__host__ __device__ aabb getMovingSphereBounds(const MovingSphere& sp) {
	aabb t0 = aabb(sp.center0 - glm::vec3(sp.radius), sp.center0 + glm::vec3(sp.radius));
	aabb t1 = aabb(sp.center1 - glm::vec3(sp.radius), sp.center1 + glm::vec3(sp.radius));
	return aabb(t0, t1);
}

__device__ bool MovingSphereHittable::ClosestIntersection(const Ray& ray, RayPayload& rec) const  {
	glm::vec3 center = glm::mix(moving_sphere->center0, moving_sphere->center1, ray.time);
	float t = _sphere_closest_intersection(ray, center, moving_sphere->radius);
	if (t >= rec.distance) return false;

	rec.material_ptr = mat_ptr;
	rec.distance = t;
	auto& sp_rec = *reinterpret_cast<MovingSphere::TraceRecord*>(&rec.payload);
	sp_rec.moving_sphere = moving_sphere;
	sp_rec.normal = (ray.at(rec.distance) - center) / moving_sphere->radius;
	return true;
}



void SphereHandle::_delete() {
	CUDA_ASSERT(cudaFree(material_ptr));
	CUDA_ASSERT(cudaFree(sphere_ptr));
	CUDA_ASSERT(cudaFree(moving_sphere_ptr));
	CUDA_ASSERT(cudaFree(hittable_ptr));

	material_ptr = nullptr;
	sphere_ptr = nullptr;
	moving_sphere_ptr = nullptr;
	hittable_ptr = nullptr;
}

SphereHandle::SphereHandle(SphereHandle&& sp) {
	bounds = sp.bounds;
	material_ptr = sp.material_ptr;
	sphere_ptr = sp.sphere_ptr;
	moving_sphere_ptr = sp.moving_sphere_ptr;
	hittable_ptr = sp.hittable_ptr;

	sp.material_ptr = nullptr;
	sp.sphere_ptr = nullptr;
	sp.moving_sphere_ptr = nullptr;
	sp.hittable_ptr = nullptr;
}

SphereHandle& SphereHandle::operator=(SphereHandle&& sp) {
	_delete();

	bounds = sp.bounds;
	material_ptr = sp.material_ptr;
	sphere_ptr = sp.sphere_ptr;
	moving_sphere_ptr = sp.moving_sphere_ptr;
	hittable_ptr = sp.hittable_ptr;

	sp.material_ptr = nullptr;
	sp.sphere_ptr = nullptr;
	sp.moving_sphere_ptr = nullptr;
	sp.hittable_ptr = nullptr;

	return *this;
}