#ifndef MATERIAL_ABSTRACT_CLASS_H
#define MATERIAL_ABSTRACT_CLASS_H

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include "cuRandom.cuh"
#include "ray_data.cuh"
#include "hittable.cuh"


// abstract class that all material classes should inherit from
class Material {
protected:
	// protected default constructor, copy constructor and copy assignment,
	//   guaranteeing that no instance of Material will ever exist outside of this class
	Material() = default;
	Material(const Material&) = default;
	Material& operator=(const Material&) = default;

public:
	__device__ virtual bool Scatter(
		const Ray& in_ray, const RayPayload& rec,
		cuRandom& rng,
		Ray& scatter_ray, glm::vec3& attenuation
	) const = 0;
};

class GeoIndependantMaterial : public Material {
protected:
	GeoIndependantMaterial() = default;
	GeoIndependantMaterial(const GeoIndependantMaterial&) = default;
	GeoIndependantMaterial& operator=(const GeoIndependantMaterial&) = default;
};

template <Geometry_t G>
class GeometryDependantMaterial : public Material {
protected:
	GeometryDependantMaterial() = default;
	GeometryDependantMaterial(const GeometryDependantMaterial&) = default;
	GeometryDependantMaterial& operator=(const GeometryDependantMaterial&) = default;
};


template <typename GeoType, typename MatType>
concept GeoDependantMat = std::derived_from<MatType, GeometryDependantMaterial<GeoType>>;

template <typename MatType>
concept GeoIndependantMat = std::derived_from<MatType, GeoIndependantMaterial>;

template <typename GeoType, typename MatType>
concept GeoAcceptableMat = GeoIndependantMat<MatType> || GeoDependantMat<GeoType, MatType>;

#endif // MATERIAL_ABSTRACT_CLASS_H //