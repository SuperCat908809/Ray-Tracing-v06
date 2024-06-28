#ifndef CUDA_TEXTURE_CLASSES_CUH
#define CUDA_TEXTURE_CLASSES_CUH

#include "texture.cuh"
#include <glm/glm.hpp>
#include <glm/gtx/component_wise.hpp>


class solid_texture : public Texture {
	glm::vec3 color{};
public:

	__device__ solid_texture() = default;
	__device__ solid_texture(glm::vec3 color) : color(color) {}

	__device__ virtual glm::vec3 value(glm::vec2 tex_coord, glm::vec3 pos) const override {
		return color;
	}
};

class checker_texture : public Texture {
	Texture* even{ nullptr };
	Texture* odd{ nullptr };
	float inv_scale{ 1.0f };
public:

	__device__ checker_texture() = default;
	__device__ checker_texture(solid_texture* c1, solid_texture* c2, float scale)
		: inv_scale(1.0f / scale),
		even(c1), odd(c2) {}

	__device__ virtual glm::vec3 value(glm::vec2 tex_coord, glm::vec3 pos) const override {
		glm::ivec3 i = glm::ivec3(pos * inv_scale);

		bool isEven = (glm::compAdd(i)) % 2 == 0;
		Texture* tex = isEven ? even : odd;

		return tex->value(tex_coord, pos);
	}
};

#endif // CUDA_TEXTURE_CLASSES_CUH //