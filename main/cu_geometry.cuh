#ifndef CU_GEOMETRY_CLASSES_H
#define CU_GEOMETRY_CLASSES_H

#include "cu_rtCommon.cuh"

// main code of geometry will be in global inline functions while I figure out the most performant method to join it all together into one program.

__host__ __device__ inline bool g_trace_sphere(Ray& ray, TraceRecord& rec, glm::vec3 origin, float radius) {
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
	return true;
}

class Hittable {
public:
	__device__ virtual ~Hittable() {};
	__device__ virtual bool ClosestIntersection(Ray& ray, TraceRecord& rec) const = 0;
};

class Sphere : public Hittable {
	glm::vec3 origin{ 0,0,0 };
	float radius{ 1 };
	Material* mat_ptr{ nullptr };

public:
	struct ConstructorParams {
		glm::vec3 origin{};
		float radius{ 1 };
		Material* mat_ptr{ nullptr };

		__host__ __device__ ConstructorParams(glm::vec3 origin, float radius, Material* mat_ptr) : origin(origin), radius(radius), mat_ptr(mat_ptr) {}
	};

	__device__ Sphere(glm::vec3 origin, float radius, Material* mat_ptr) : origin(origin), radius(radius), mat_ptr(mat_ptr) {}
	__device__ Sphere(const ConstructorParams& p) : origin(p.origin), radius(p.radius), mat_ptr(p.mat_ptr) {}

	__device__ bool ClosestIntersection(Ray& ray, TraceRecord& rec) const {
		if (g_trace_sphere(ray, rec, origin, radius)) {
			rec.mat_ptr = mat_ptr;
			return true;
		}
		return false;
	}
};

class HittableList : public Hittable {
	Hittable** objects{};
	int object_count{};

public:
	__device__ HittableList() = delete;
	__device__ HittableList(const HittableList&) = delete;
	__device__ HittableList& operator=(const HittableList&) = delete;

	__device__ HittableList(Hittable** objects, int object_count) : objects(objects), object_count(object_count) {}

	__device__ bool ClosestIntersection(Ray& ray, TraceRecord& rec) const {
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

#endif // CU_GEOMETRY_CLASSES_H //