#ifndef RENDERER_CLASS_H
#define RENDERER_CLASS_H

#include "cu_rtCommon.cuh"
#include "cu_Cameras.cuh"
#include "cu_Geometry.cuh"

class Renderer {

	uint32_t render_width{}, render_height{}, samples_per_pixel{};
	PinholeCamera cam{};

	// cuda memory
	HittableList<Sphere>* d_sphere_list{};
	glm::vec4* d_output_buffer{};
	curandState_t* d_random_states{};

public:

	Renderer() = delete;
	Renderer(const Renderer&) = delete;
	Renderer& operator=(const Renderer&) = delete;
	Renderer(
		uint32_t render_width, uint32_t render_height, uint32_t samples_per_pixel,
		const PinholeCamera& cam,
		const HittableList<Sphere>* h_sphere_list
	);
	~Renderer();

	void DownloadRenderbuffer(glm::vec4* host_dst) const;
	void Render();
};

#endif // RENDERER_CLASS_H //