#ifndef MATERIAL_ABSTRACT_CLASS_H
#define MATERIAL_ABSTRACT_CLASS_H

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "ray_data.cuh"


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
		curandState_t* random_state,
		Ray& scatter_ray, glm::vec3& attenuation
	) const = 0;
};

#endif // MATERIAL_ABSTRACT_CLASS_H //