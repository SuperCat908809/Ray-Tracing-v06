#ifndef CU_GEOMETRY_CLASSES_H
#define CU_GEOMETRY_CLASSES_H

#include "cu_rtCommon.cuh"
#include <vector>

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
	rec.n = glm::normalize(ray.at(t) - origin);
	if (glm::dot(ray.d, rec.n) > 0) {
		// ray is inside the sphere
		rec.n = -rec.n;
		rec.hit_backface = true;
	}
	return true;
}

struct Sphere {
	glm::vec3 origin{ 0,0,0 };
	float radius{ 1 };
	Material* mat_ptr{ nullptr };

	__host__ __device__ Sphere(glm::vec3 origin, float radius, Material* mat_ptr) : origin(origin), radius(radius), mat_ptr(mat_ptr) {}

	__host__ __device__ bool ClosestIntersection(Ray& ray, TraceRecord& rec) const {
		if (g_trace_sphere(ray, rec, origin, radius)) {
			rec.mat_ptr = mat_ptr;
			return true;
		}
		return false;
	}
};

template <typename T>
struct HittableList {
private:
	T* objects{};
	int object_count{};

public:
	__host__ __device__ HittableList() = delete;
	__host__ __device__ HittableList(const HittableList<Sphere>&) = delete;
	__host__ __device__ HittableList& operator=(const HittableList&) = delete;
	__host__ HittableList(std::vector<T> object_data) {
		CUDA_ASSERT(cudaMalloc(&objects, sizeof(T) * object_data.size()));
		CUDA_ASSERT(cudaMemcpy(objects, object_data.data(), sizeof(T) * object_data.size(), cudaMemcpyHostToDevice));
		object_count = object_data.size();
	}
	__host__ ~HittableList() {
		CUDA_ASSERT(cudaFree(objects));
	}

	__device__ bool ClosestIntersection(Ray& ray, TraceRecord& rec) const {
		bool hit_any{ false };

		for (int i = 0; i < object_count; i++) {
			hit_any |= objects[i].ClosestIntersection(ray, rec);
		}
		// rec only gets updated when an intersection has been found.
		// we want to discard the last rec if a closer one is found.
		// hence passing rec itself is ok since a further intersection would be disgarded for a closer one.

		return hit_any;
	}
};

#endif // CU_GEOMETRY_CLASSES_H //