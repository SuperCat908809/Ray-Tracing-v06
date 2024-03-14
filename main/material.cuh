#ifndef MATERIAL_ABSTRACT_CLASS_H
#define MATERIAL_ABSTRACT_CLASS_H

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include "ray_data.cuh"
#include "cuRandom.cuh"


// abstract class that all material classes should inherit from
class Material {
protected:
	// protected default constructor, copy constructor and copy assignment,
	//   guaranteeing that no instance of Material will ever exist outside of this class
	Material() = default;
	Material(const Material&) = default;
	Material& operator=(const Material&) = default;

public:
	__device__ virtual ~Material() {};
	__device__ virtual bool Scatter(
		const Ray& in_ray, const TraceRecord& rec,
		cuRandom& rng,
		Ray& scatter_ray, glm::vec3& attenuation
	) const = 0;
};

#endif // MATERIAL_ABSTRACT_CLASS_H //