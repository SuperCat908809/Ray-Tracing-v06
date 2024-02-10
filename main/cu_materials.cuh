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


__device__ inline bool g_scatter_metal(const Ray& ray, const TraceRecord& rec, curandState_t* random_state, Ray& scatter_ray, glm::vec3& attenuation, glm::vec3 albedo, float fuzz) {
	glm::vec3 scatter_dir = glm::reflect(ray.d, rec.n) + fuzz * RND_IN_SPHERE;
	scatter_ray = Ray(ray.at(rec.t) + scatter_dir * 0.0003f, scatter_dir);
	attenuation = albedo;
	return true;
}

struct MetalMaterial {
	glm::vec3 albedo{ 1.0f };
	float fuzz{ 0.1f };

	__host__ __device__ MetalMaterial() = default;
	__host__ __device__ MetalMaterial(glm::vec3 albedo, float fuzz) : albedo(albedo), fuzz(fuzz) {}

	__device__ bool Scatter(const Ray& ray, const TraceRecord& rec, curandState_t* random_state, Ray& scatter_ray, glm::vec3& attenuation) const {
		return g_scatter_metal(ray, rec, random_state, scatter_ray, attenuation, albedo, fuzz);
	}
};


__device__ inline float reflectance(float cos_theta, float ior_ratio) {
	// use Schlick's approximation for reflectance
	float r0 = (1 - ior_ratio) / (1 + ior_ratio);
	r0 *= r0;
	return r0 + (1 - r0) * powf(1 - cos_theta, 5.0f);
}

__device__ inline bool g_scatter_dielectric(const Ray& ray, const TraceRecord& rec, curandState_t* random_state, Ray& scatter_ray, glm::vec3& attenuation, glm::vec3 albedo, float ior) {
	float ior_ratio = rec.hit_backface ? ior : 1 / ior;

	glm::vec3 unit_dir = glm::normalize(ray.d);
	float cos_theta = fminf(glm::dot(-unit_dir, rec.n), 1.0f);
	float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
	float reflect_prob = reflectance(cos_theta, ior_ratio);

	glm::vec3 scatter_dir{};

	if (ior_ratio * sin_theta > 1.0f || reflect_prob > RND) {
		scatter_dir = glm::reflect(unit_dir, rec.n);
	}
	else {
		scatter_dir = glm::refract(unit_dir, rec.n, ior_ratio);
	}

	scatter_ray = Ray(ray.at(rec.t) + scatter_dir * 0.0003f, scatter_dir);
	attenuation = albedo;
	return true;
}

struct DielectricMaterial {
	glm::vec3 albedo{ 1.0f };
	float ior{ 1.333f };

	__host__ __device__ DielectricMaterial() = default;
	__host__ __device__ DielectricMaterial(glm::vec3 albedo, float ior) : albedo(albedo), ior(ior) {}

	__device__ bool Scatter(const Ray& ray, const TraceRecord& rec, curandState_t* random_state, Ray& scatter_ray, glm::vec3& attenuation) const {
		return g_scatter_dielectric(ray, rec, random_state, scatter_ray, attenuation, albedo, ior);
	}
};



// abstract class that all material structs should inherit from
struct Material {

	__device__ virtual bool Scatter(
		const Ray& ray, const TraceRecord& rec,
		curandState_t* random_state,
		Ray& scatter_ray, glm::vec3& attenuation
	) const = 0;
};

struct LambertianAbstract : public Material {
	glm::vec3 albedo{ 1.0f };

	__device__ LambertianAbstract() = default;
	__device__ LambertianAbstract(glm::vec3 albedo) : albedo(albedo) {}

	__device__ virtual bool Scatter(
		const Ray& ray, const TraceRecord& rec,
		curandState_t* random_state,
		Ray& scatter_ray, glm::vec3& attenuation
	) const override {
		return g_scatter_lambertian(ray, rec, random_state, scatter_ray, attenuation, albedo);
	}
};

struct MetalAbstract : public Material {
	glm::vec3 albedo{ 1.0f };
	float fuzz{ 0.1f };

	__device__ MetalAbstract() = default;
	__device__ MetalAbstract(glm::vec3 albedo, float fuzz) : albedo(albedo), fuzz(fuzz) {}

	__device__ virtual bool Scatter(
		const Ray& ray, const TraceRecord& rec,
		curandState_t* random_state,
		Ray& scatter_ray, glm::vec3& attenuation
	) const override {
		return g_scatter_metal(ray, rec, random_state, scatter_ray, attenuation, albedo, fuzz);
	}
};

struct DielectricAbstract : public Material {
	glm::vec3 albedo{ 1.0f };
	float ior{ 1.333f };

	__device__ DielectricAbstract() = default;
	__device__ DielectricAbstract(glm::vec3 albedo, float ior) : albedo(albedo), ior(ior) {}

	__device__ virtual bool Scatter(
		const Ray& ray, const TraceRecord& rec,
		curandState_t* random_state,
		Ray& scatter_ray, glm::vec3& attenuation
	) const override {
		return g_scatter_dielectric(ray, rec, random_state, scatter_ray, attenuation, albedo, ior);
	}
};

#endif // CU_MATERIAL_CLASSES_H //