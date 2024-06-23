#ifndef CUDA_TEXTURE_CLASS_CUH
#define CUDA_TEXTURE_CLASS_CUH

#include <cuda_runtime.h>
#include <glm/glm.hpp>


class Texture {
protected:

	__device__ Texture() = default;
	__device__ Texture(const Texture&) = default;
	__device__ Texture& operator=(const Texture&) = default;

public:

	__device__ virtual ~Texture() = default;

	__device__ virtual glm::vec3 value(glm::vec2 tex_coord, glm::vec3 pos) const = 0;
};

#endif // CUDA_TEXTURE_CLASS_CUH //