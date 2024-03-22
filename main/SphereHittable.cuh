#ifndef CU_GEOMETRY_CLASSES_H
#define CU_GEOMETRY_CLASSES_H

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include "ray_data.cuh"
#include "hittable.cuh"


__device__ inline bool _closest_sphere_intersection(const Ray& ray, TraceRecord& rec, glm::vec3 center, float radius) {
	glm::vec3 oc = ray.o - center;

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
	glm::vec3 normal = (ray.at(rec.t) - center) / radius; // a negative radius will flip the normal as intended
	rec.set_face_normal(ray, normal);
	return true;
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


class SphereHittable : public Hittable {
	glm::vec3 center;
	float radius;
	Material* mat_ptr;

public:

	__device__ SphereHittable(glm::vec3 origin, float radius, Material* mat_ptr) : center(origin), radius(radius), mat_ptr(mat_ptr) {}
	__device__ virtual bool ClosestIntersection(const Ray& ray, TraceRecord& rec) const override {
		if (_closest_sphere_intersection(ray, rec, center, radius)) {
			rec.mat_ptr = mat_ptr;
			return true;
		}
		return false;
	}
	__device__ virtual aabb bounding_box() const {
		glm::vec3 r(radius);
		return aabb(center - r, center + r);
	}
};

class MovingSphereHittable : public Hittable {
	glm::vec3 center0;
	glm::vec3 center1;
	float radius;
	Material* mat_ptr;

public:

	__device__ MovingSphereHittable(glm::vec3 center0, glm::vec3 center1, float radius, Material* mat_ptr) : center0(center0), center1(center1), radius(radius), mat_ptr(mat_ptr) {}
	__device__ virtual bool ClosestIntersection(const Ray& ray, TraceRecord& rec) const override {
		glm::vec3 center = glm::mix(center0, center1, ray.time);
		if (_closest_sphere_intersection(ray, rec, center, radius)) {
			rec.mat_ptr = mat_ptr;
			return true;
		}
		return false;
	}
	__device__ virtual aabb bounding_box() const {
		glm::vec3 r(radius);
		aabb a(center0 - r, center0 + r);
		aabb b(center1 - r, center1 + r);
		return aabb(a, b);
	}
};

#endif // CU_GEOMETRY_CLASSES_H //