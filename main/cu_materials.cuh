#ifndef CU_MATERIAL_CLASSES_H
#define CU_MATERIAL_CLASSES_H

#include "cu_rtCommon.cuh"

// main code of materials will be in global inline functions while I figure out the most performant method to join it all together into one program.

__device__ inline bool g_scatter_lambertian(const Ray& ray, const TraceRecord& rec, curandState_t* random_state, Ray& scatter_ray, glm::vec3& attenuation, glm::vec3 albedo) {
	glm::vec3 ray_dir = rec.n + RND_ON_SPHERE;
	if (glm::near_zero(ray_dir)) return false;

	scatter_ray = Ray(ray.at(rec.t) + ray_dir * 0.0003f, ray_dir);
	attenuation = albedo;
	return true;
}

struct LambertianMaterial {
	glm::vec3 albedo{ 0.5f };

	__host__ __device__ LambertianMaterial() = default;
	__host__ __device__ LambertianMaterial(glm::vec3 albedo) : albedo(albedo) {}

	__device__ bool Scatter(const Ray& ray, const TraceRecord& rec, curandState_t* random_state, Ray& scatter_ray, glm::vec3& attenuation) const {
		return g_scatter_lambertian(ray, rec, random_state, scatter_ray, attenuation, albedo);
	}
};


__device__ inline bool g_scatter_metal(const Ray& ray, const TraceRecord& rec, curandState_t* random_state, Ray& scatter_ray, glm::vec3& attenuation, glm::vec3 albedo) {
	glm::vec3 reflected = glm::reflect(ray.d, rec.n);
	scatter_ray = Ray(ray.at(rec.t) + reflected * 0.0003f, reflected);
	attenuation = albedo;
	return true;
}

struct MetalMaterial {
	glm::vec3 albedo{ 1.0f };

	__host__ __device__ MetalMaterial() = default;
	__host__ __device__ MetalMaterial(glm::vec3 albedo) : albedo(albedo) {}

	__device__ bool Scatter(const Ray& ray, const TraceRecord& rec, curandState_t* random_state, Ray& scatter_ray, glm::vec3& attenuation) const {
		return g_scatter_metal(ray, rec, random_state, scatter_ray, attenuation, albedo);
	}
};

#endif // CU_MATERIAL_CLASSES_H //