#ifndef RENDERER_CLASS_H
#define RENDERER_CLASS_H

#include <inttypes.h>
#include <glm/glm.hpp>


class Hittable;
class MotionBlurCamera;
class cuRandom;

class Renderer {

	struct M {
		uint32_t render_width{}, render_height{};
		uint32_t samples_per_pixel{}, max_depth{};
		const MotionBlurCamera* cam{};

		// cuda memory
		const Hittable* d_world_ptr{};
		glm::vec4* d_output_buffer;
		cuRandom* rngs;
	} m;

	void _delete();

	Renderer(M m) : m(std::move(m)) {}

	Renderer(const Renderer&) = delete;
	Renderer& operator=(const Renderer&) = delete;

public:

	~Renderer();
	Renderer(Renderer&& other);
	Renderer& operator=(Renderer&& other);

	static Renderer MakeRenderer(
		uint32_t render_width, uint32_t render_height,
		uint32_t samples_per_pixel, uint32_t max_depth,
		const MotionBlurCamera* cam,
		const Hittable* d_world_ptr
	);

	void Render();
	void DownloadRenderbuffer(glm::vec4* host_dst) const;
};

#endif // RENDERER_CLASS_H //