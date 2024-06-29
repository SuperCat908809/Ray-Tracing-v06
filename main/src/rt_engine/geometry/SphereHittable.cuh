#ifndef CU_GEOMETRY_CLASSES_H
#define CU_GEOMETRY_CLASSES_H

#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "../../utilities/cuda_utilities/cuda_utils.cuh"

#include "../ray_data.cuh"
#include "aabb.cuh"
#include "hittable.cuh"
#include "../shaders/material.cuh"


__device__ inline float _sphere_closest_intersection(const Ray& ray, glm::vec3 center, float radius) {
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


class Sphere : public Geometry {

	friend class SphereHittable;
	struct TraceRecord {
		const Sphere* sphere{};
		glm::vec3 normal{};
	};
	static_assert(sizeof(Sphere::TraceRecord) <= sizeof(RayPayload::Payload),
		"Ray payload is too small to fit SphereHittable::TraceRecord");
	static_assert(alignof(Sphere::TraceRecord) <= alignof(RayPayload::Payload),
		"Ray payload alignment is too small to fit SphereHittable::TraceRecord");

public:
	glm::vec3 center;
	float radius;


	__host__ __device__ Sphere() = default;
	__host__ __device__ Sphere(glm::vec3 center, float radius) : center(center), radius(radius) {};

	__host__ __device__ static glm::vec3 getNormal(const Ray& ray, const RayPayload& rec) {
		auto& sp_rec = *reinterpret_cast<const Sphere::TraceRecord*>(&rec.payload);
		return sp_rec.normal;
		//const Sphere& sp = *sp_rec.sphere;
		//glm::vec3 intersect_pos = ray.at(rec.distance);
		//glm::vec3 normal = (intersect_pos - sp.center) / sp.radius;
		//return normal;
	}
};

__host__ __device__ inline aabb getSphereBounds(const Sphere& sp) {
	return aabb(sp.center - glm::vec3(sp.radius), sp.center + glm::vec3(sp.radius));
}

class SphereHittable : public Hittable {
	const Sphere* sphere;
	const Material* mat_ptr;

public:

	template <typename MatType> requires GeoAcceptableMat<Sphere, MatType>
	__device__ SphereHittable(const Sphere* sphere, const MatType* mat_ptr) : sphere(sphere), mat_ptr(mat_ptr) {}

	__device__ virtual bool ClosestIntersection(const Ray& ray, RayPayload& rec) const override {
		float t = _sphere_closest_intersection(ray, sphere->center, sphere->radius);
		if (t >= rec.distance) return false;

		rec.material_ptr = mat_ptr;
		rec.distance = t;
		auto& sp_rec = *reinterpret_cast<Sphere::TraceRecord*>(&rec.payload);
		sp_rec.sphere = sphere;
		sp_rec.normal = (ray.at(rec.distance) - sphere->center) / sphere->radius;
		return true;
	}
};


class MovingSphere : public Geometry {

	friend class MovingSphereHittable;
	struct TraceRecord {
		const MovingSphere* moving_sphere{};
		glm::vec3 normal{};
	};
	static_assert(sizeof(MovingSphere::TraceRecord) <= sizeof(RayPayload::Payload),
		"Ray payload is too small to fit MovingSphereHittable::TraceRecord");
	static_assert(alignof(MovingSphere::TraceRecord) <= alignof(RayPayload::Payload),
		"Ray payload alignment is too small to fit MovingSphereHittable::TraceRecord");

public:
	glm::vec3 center0;
	glm::vec3 center1;
	float radius;

	__host__ __device__ MovingSphere() = default;
	__host__ __device__ MovingSphere(glm::vec3 center0, glm::vec3 center1, float radius) 
		: center0(center0), center1(center1), radius(radius) {}

	__host__ __device__ static glm::vec3 getNormal(const Ray& ray, const RayPayload& rec) {
		auto& sp_rec = *reinterpret_cast<const MovingSphere::TraceRecord*>(&rec.payload);
		return sp_rec.normal;
		//const MovingSphere& sp = *sp_rec.moving_sphere;
		//glm::vec3 intersect_pos = ray.at(rec.distance);
		//glm::vec3 time_sliced_center = glm::mix(sp.center0, sp.center1, ray.time);
		//glm::vec3 normal = (intersect_pos - time_sliced_center) / sp.radius;
		//return normal;
	}
};

__host__ __device__ inline aabb getMovingSphereBounds(const MovingSphere& sp) {
	aabb t0 = aabb(sp.center0 - glm::vec3(sp.radius), sp.center0 + glm::vec3(sp.radius));
	aabb t1 = aabb(sp.center1 - glm::vec3(sp.radius), sp.center1 + glm::vec3(sp.radius));
	return aabb(t0, t1);
}

class MovingSphereHittable : public Hittable {
	const MovingSphere* moving_sphere;
	const Material* mat_ptr;

public:

	template <typename MatType> requires GeoAcceptableMat<MovingSphere, MatType>
	__device__ MovingSphereHittable(const MovingSphere* moving_sphere, const MatType* mat_ptr) : moving_sphere(moving_sphere), mat_ptr(mat_ptr) {}

	__device__ virtual bool ClosestIntersection(const Ray& ray, RayPayload& rec) const override {
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
};


class SphereHandle {

	aabb bounds;
	Material* material_ptr{};
	Sphere* sphere_ptr{};
	MovingSphere* moving_sphere_ptr{};
	Hittable* hittable_ptr{};
	// only one of the two spheres above will be instantiated

	SphereHandle() = default;

	void _delete() {
		CUDA_ASSERT(cudaFree(material_ptr));
		CUDA_ASSERT(cudaFree(sphere_ptr));
		CUDA_ASSERT(cudaFree(moving_sphere_ptr));
		CUDA_ASSERT(cudaFree(hittable_ptr));

		material_ptr = nullptr;
		sphere_ptr = nullptr;
		moving_sphere_ptr = nullptr;
		hittable_ptr = nullptr;
	}

	SphereHandle(const SphereHandle&) = delete;
	SphereHandle& operator=(const SphereHandle&) = delete;

public:

	SphereHandle(SphereHandle&& sp) {
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

	~SphereHandle() {
		_delete();
	}

	template <typename MatType> requires GeoAcceptableMat<Sphere, MatType>
	static SphereHandle MakeSphere(const Sphere& sphere, MatType* mat_ptr) {
		SphereHandle sp{};
		sp.bounds = getSphereBounds(sphere);
		sp.material_ptr = mat_ptr;
		sp.sphere_ptr = newOnDevice<Sphere>(sphere);
		sp.moving_sphere_ptr = nullptr;
		sp.hittable_ptr = newOnDevice<SphereHittable>(sp.sphere_ptr, mat_ptr);
		return sp;
	}

	template <typename MatType> requires GeoAcceptableMat<MovingSphere, MatType>
	static SphereHandle MakeMovingSphere(const MovingSphere& sphere, MatType* mat_ptr) {
		SphereHandle sp{};
		sp.bounds = getMovingSphereBounds(sphere);
		sp.material_ptr = mat_ptr;
		sp.sphere_ptr = nullptr;
		sp.moving_sphere_ptr = newOnDevice<MovingSphere>(sphere);
		sp.hittable_ptr = newOnDevice<MovingSphereHittable>(sp.moving_sphere_ptr, mat_ptr);
		return sp;
	}

	const Hittable* getHittablePtr() const { return hittable_ptr; }
	aabb getBounds() const { return bounds; }
};

#endif // CU_GEOMETRY_CLASSES_H //