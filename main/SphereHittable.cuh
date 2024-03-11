#ifndef CU_GEOMETRY_CLASSES_H
#define CU_GEOMETRY_CLASSES_H

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include "ray_data.cuh"
#include "hittable.cuh"


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

class SphereHittable : public Hittable {
	Sphere sphere;

public:

	__device__ SphereHittable(const Sphere& sphere) : sphere(sphere) {}
	__device__ SphereHittable(glm::vec3 origin, float radius, Material* mat_ptr) : sphere(origin, radius, mat_ptr) {}
	__device__ virtual bool ClosestIntersection(const Ray& ray, TraceRecord& rec) const override {
		return sphere.ClosestIntersection(ray, rec);
	}
};

#endif // CU_GEOMETRY_CLASSES_H //