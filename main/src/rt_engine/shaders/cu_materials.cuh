#ifndef CU_MATERIAL_CLASSES_H
#define CU_MATERIAL_CLASSES_H

#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "../../utilities/glm_utils.h"
#include "../../utilities/cuda_utilities/cuRandom.cuh"

#include "../ray_data.cuh"
#include "material.cuh"
#include "texture.cuh"
#include "cu_Textures.cuh"


template <Geometry_t GeoType>
class LambertianTexture : public GeometryDependantMaterial<GeoType> {
	checker_texture tex;
	solid_texture c1, c2;
public:

	__device__ LambertianTexture() = default;
	__device__ LambertianTexture(glm::vec3 c1, glm::vec3 c2, float scale) {
		this->c1 = solid_texture(c1);
		this->c2 = solid_texture(c2);
		this->tex = checker_texture(&this->c1, &this->c2, scale);
	}
	__device__ virtual bool Scatter(
		const Ray& in_ray, const RayPayload& rec,
		cuRandom& rng,
		Ray& scatter_ray, glm::vec3& attenuation
	) const override {
		glm::vec3 normal = GeoType::getNormal(in_ray, rec);
		glm::vec3 ray_dir = normal + glm::cuRandomOnUnit<3>(rng);
		if (glm::near_zero(ray_dir)) return false;

		scatter_ray = Ray(in_ray.at(rec.distance), ray_dir, in_ray.time);
		attenuation = tex.value(glm::vec2(0, 0), in_ray.at(rec.distance));
		return true;
	}
};


template <Geometry_t GeoType>
class LambertianAbstract : public GeometryDependantMaterial<GeoType> {
	glm::vec3 albedo{ 1.0f };

public:
	__device__ LambertianAbstract() = default;
	__device__ LambertianAbstract(glm::vec3 albedo) : albedo(albedo) {}

	__device__ virtual bool Scatter(
		const Ray& in_ray, const RayPayload& rec,
		cuRandom& rng,
		Ray& scatter_ray, glm::vec3& attenuation
	) const override {
		glm::vec3 normal = GeoType::getNormal(in_ray, rec);
		glm::vec3 ray_dir = normal + glm::cuRandomOnUnit<3>(rng);
		if (glm::near_zero(ray_dir)) return false;

		scatter_ray = Ray(in_ray.at(rec.distance), ray_dir, in_ray.time);
		attenuation = albedo;
		return true;
	}
};


template <Geometry_t GeoType>
class MetalAbstract : public GeometryDependantMaterial<GeoType> {
	glm::vec3 albedo{ 1.0f };
	float fuzz{ 0.0f };

public:
	__device__ MetalAbstract() = default;
	__device__ MetalAbstract(glm::vec3 albedo, float fuzz) : albedo(albedo), fuzz(fuzz) {}

	__device__ virtual bool Scatter(
		const Ray& in_ray, const RayPayload& rec,
		cuRandom& rng,
		Ray& scatter_ray, glm::vec3& attenuation
	) const override {
		glm::vec3 normal = GeoType::getNormal(in_ray, rec);
		glm::vec3 scatter_dir = glm::reflect(in_ray.d, normal) + fuzz * glm::cuRandomOnUnit<3>(rng);

		if (glm::dot(scatter_dir, normal) < 0 || glm::near_zero(scatter_dir)) {
			// reflection points into surface
			//   or reflection length too small to be useful
			// ray absorbed
			return false;
		}

		scatter_ray = Ray(in_ray.at(rec.distance), scatter_dir, in_ray.time);
		attenuation = albedo;
		return true;
	}
};


__device__ inline float reflectance(float cos_theta, float ior_ratio) {
	// use Schlick's approximation for reflectance
	float r0 = (1 - ior_ratio) / (1 + ior_ratio);
	r0 = r0 * r0;
	return r0 + (1 - r0) * powf(1 - cos_theta, 5.0f);
}

template <Geometry_t GeoType>
class DielectricAbstract : public GeometryDependantMaterial<GeoType> {
	glm::vec3 albedo{ 1.0f };
	float ior{ 1.333f };

public:
	__device__ DielectricAbstract() = default;
	__device__ DielectricAbstract(glm::vec3 albedo, float ior) : albedo(albedo), ior(ior) {}

	__device__ virtual bool Scatter(
		const Ray& in_ray, const RayPayload& rec,
		cuRandom& rng,
		Ray& scatter_ray, glm::vec3& attenuation
	) const override {
		glm::vec3 normal = GeoType::getNormal(in_ray, rec);
		bool hit_backface = isBackfacing(in_ray, normal);
		if (hit_backface) normal = -normal;
		float ior_ratio = hit_backface ? ior : 1 / ior;


		glm::vec3 unit_dir = glm::normalize(in_ray.d);
		float cos_theta = fminf(glm::dot(-unit_dir, normal), 1.0f);
		float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
		float reflect_prob = reflectance(cos_theta, ior_ratio);

		glm::vec3 scatter_dir{};

		if (ior_ratio * sin_theta > 1.0f || reflect_prob > rng.next()) {
			scatter_dir = glm::reflect(unit_dir, normal);
		}
		else {
			scatter_dir = glm::refract(unit_dir, normal, ior_ratio);
		}

		scatter_ray = Ray(in_ray.at(rec.distance), scatter_dir, in_ray.time);
		attenuation = albedo;
		return true;
	}
};

#endif // CU_MATERIAL_CLASSES_H //