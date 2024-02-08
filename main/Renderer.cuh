#ifndef RENDERER_CLASS_H
#define RENDERER_CLASS_H

#include "cu_rtCommon.cuh"
#include "cu_Cameras.cuh"
#include "cu_Geometry.cuh"

class Renderer {

	uint32_t render_width{}, render_height{};
	PinholeCamera cam{};

	// cuda memory
	SphereList* d_sphere_list{};
	glm::vec4* d_output_buffer{};

public:

	Renderer() = delete;
	Renderer(const Renderer&) = delete;
	Renderer& operator=(const Renderer&) = delete;
	Renderer(
		uint32_t render_width, uint32_t render_height,
		const PinholeCamera& cam,
		const SphereList* h_sphere_list
	);
	~Renderer();

	void DownloadRenderbuffer(glm::vec4* host_dst) const;
	void Render();
};

#endif // RENDERER_CLASS_H //