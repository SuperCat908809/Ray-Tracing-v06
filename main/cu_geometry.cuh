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
	return true;
}

struct Sphere {
	glm::vec3 origin{ 0,0,0 };
	float radius{ 1 };

	__host__ __device__ bool Trace(Ray& ray, TraceRecord& rec) const {
		return g_trace_sphere(ray, rec, origin, radius);
	}
};

struct SphereList {
	glm::vec4* spheres{};
	int sphere_count{};

	__host__ SphereList(std::vector<glm::vec4> sphere_data) {
		CUDA_ASSERT(cudaMalloc(&spheres, sizeof(glm::vec4) * sphere_data.size()));
		CUDA_ASSERT(cudaMemcpy(spheres, sphere_data.data(), sizeof(glm::vec4) * sphere_data.size(), cudaMemcpyHostToDevice));
		sphere_count = sphere_data.size();
	}
	__host__ ~SphereList() {
		CUDA_ASSERT(cudaFree(spheres));
	}

	__device__ bool Trace(Ray& ray, TraceRecord& rec) const {
		bool hit_any{ false };

		for (int i = 0; i < sphere_count; i++) {
			glm::vec4 s = spheres[i];
			glm::vec3 origin = glm::vec3(s);
			float radius = s.w;
			hit_any |= g_trace_sphere(ray, rec, origin, radius);
		}
		// rec only gets updated when an intersection has been found.
		// we want to discard the last rec if a closer one is found.
		// hence passing rec itself is ok since a further intersection would be disgarded for a closer one.

		return hit_any;
	}
};

#endif // CU_GEOMETRY_CLASSES_H //