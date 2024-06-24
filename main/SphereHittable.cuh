#ifndef CU_GEOMETRY_CLASSES_H
#define CU_GEOMETRY_CLASSES_H

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include "ray_data.cuh"
#include "hittable.cuh"


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

#if 0
	rec.t = t;
	glm::vec3 normal = (ray.at(rec.t) - sp.center) / sp.radius; // a negative radius will flip the normal as intended
	rec.set_face_normal(ray, normal);
	return true;
#endif
}

#if 0
class Sphere {
	glm::vec3 origin{ 0,0,0 };
	float radius{ 1 };
	Material* mat_ptr{ nullptr };

public:

	__host__ __device__ Sphere(glm::vec3 origin, float radius, Material* mat_ptr) : origin(origin), radius(radius), mat_ptr(mat_ptr) {}

	__device__ bool ClosestIntersection(const Ray& ray, TraceRecord& rec) const {
		glm::vec3 oc = ray.o - origin;

		float a = glm::dot(ray.d, ray.d);
		float hb = glm::dot(ray.d, oc);
		float c = glm::dot(oc, oc) - radius * radius;
		float d = hb * hb - a * c;
		if (d <= 0) return false;

		d = sqrtf(d);
		float t = (-hb - d) / a;
		if (t < 0.0f || t > rec.t) {
			t = (-hb + d) / a;
			if (t < 0.0f || t > rec.t)
				return false;
		}

		rec.t = t;
		glm::vec3 normal = (ray.at(rec.t) - origin) / radius; // a negative radius will flip the normal as intended
		rec.set_face_normal(ray, normal);
		rec.mat_ptr = mat_ptr;
		return true;
	}
};
#endif


class Sphere : public Geometry {

	friend class SphereHittable;
	struct TraceRecord {
		const Sphere* sphere{};
	};
	static_assert(sizeof(Sphere::TraceRecord) <= sizeof(RayPayload::payload),
		"Ray payload is too small to fit SphereHittable::TraceRecord");
	static_assert(alignof(Sphere::TraceRecord) <= alignof(RayPayload),
		"Ray payload alignment is too small to fit SphereHittable::TraceRecord");

public:
	glm::vec3 center;
	float radius;


	__host__ __device__ Sphere() = default;
	__host__ __device__ Sphere(glm::vec3 center, float radius) : center(center), radius(radius) {};

	__host__ __device__ static glm::vec3 getNormal(const Ray& ray, const RayPayload& rec) {
		auto& sp_rec = *reinterpret_cast<const Sphere::TraceRecord*>(&rec.payload);
		const Sphere& sp = *sp_rec.sphere;
		glm::vec3 intersect_pos = ray.at(rec.distance);
		glm::vec3 normal = (intersect_pos - sp.center) / sp.radius;
		return normal;
	}
};

class SphereHittable : public Hittable {
	const Sphere* sphere;
	const Material* mat_ptr;

public:

	template <typename MatType> requires GeoAcceptable<Sphere, MatType>
	__device__ SphereHittable(const Sphere* sphere, const MatType* mat_ptr) : sphere(sphere), mat_ptr(mat_ptr) {}
	__device__ virtual bool ClosestIntersection(const Ray& ray, RayPayload& rec) const override {
		float t = _sphere_closest_intersection(ray, sphere->center, sphere->radius);
		if (t >= rec.distance) return false;

		rec.material_ptr = mat_ptr;
		rec.distance = t;
		auto& sp_rec = *reinterpret_cast<Sphere::TraceRecord*>(&rec.payload);
		sp_rec.sphere = sphere;
		return true;
	}
};



/*

I realise that constructing a bvh hierarchy must be done on the device, thus making the bounding_box function only accessible on the device is a problem.
Perhaps I can re-work everything such that during construction, like through a factory object, all the data pertaining to be device side object is available on the host, 
then whenever I need something like its bounding box then I can easily access it and once everything is done host-side, every device side object will be constructed and 
all the necessary pointer passing will occur such that everything points to the correct data.


Hittable should only provide the interface for intersectable geometry and doesn't have the responsibility of returning the AABB.
It should be up to the host to provide the AABB and any other data not related directly to rendering.
Generating an AABB should be in the scene construction step so the host should store whatever it needs, either a full copy of the Hittable or a pointer to fetch it if required, 
which will be used to provide the requested data (AABB and such).

Perhaps I can make a second abstract class that provides these accessors, specifically for host side code.

*/


class MovingSphere : public Geometry {

	friend class MovingSphereHittable;
	struct TraceRecord {
		const MovingSphere* moving_sphere{};
	};
	static_assert(sizeof(MovingSphere::TraceRecord) <= sizeof(RayPayload::payload),
		"Ray payload is too small to fit MovingSphereHittable::TraceRecord");
	static_assert(alignof(MovingSphere::TraceRecord) <= alignof(RayPayload),
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
		const MovingSphere& sp = *sp_rec.moving_sphere;
		glm::vec3 intersect_pos = ray.at(rec.distance);
		glm::vec3 time_sliced_center = glm::mix(sp.center0, sp.center1, ray.time);
		glm::vec3 normal = (intersect_pos - time_sliced_center) / sp.radius;
		return normal;
	}
};

class MovingSphereHittable : public Hittable {
	const MovingSphere* moving_sphere;
	const Material* mat_ptr;

public:

	template <typename MatType> requires GeoAcceptable<MovingSphere, MatType>
	__device__ MovingSphereHittable(const MovingSphere* moving_sphere, const MatType* mat_ptr) : moving_sphere(moving_sphere), mat_ptr(mat_ptr) {}
	__device__ virtual bool ClosestIntersection(const Ray& ray, RayPayload& rec) const override {
		glm::vec3 center = glm::mix(moving_sphere->center0, moving_sphere->center1, ray.time);
		float t = _sphere_closest_intersection(ray, center, moving_sphere->radius);
		if (t >= rec.distance) return false;

		rec.material_ptr = mat_ptr;
		rec.distance = t;
		auto& sp_rec = *reinterpret_cast<MovingSphere::TraceRecord*>(&rec.payload);
		sp_rec.moving_sphere = moving_sphere;
		return true;
	}
};

#endif // CU_GEOMETRY_CLASSES_H //