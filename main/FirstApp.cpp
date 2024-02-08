#include "FirstApp.h"
#include <string>
#include <stb/stb_image_write.h>
#include <glm/glm.hpp>

FirstApp::FirstApp() {
	render_width = 1280;
	render_height = 720;

	glm::vec3 lookfrom(0, 1, 4);
	glm::vec3 lookat(0, 1, 0);
	glm::vec3 up(0, 1, 0);
	float fov = 90.0f;
	float aspect = render_width / (float)render_height;
	cam = PinholeCamera(lookfrom, lookat, up, fov, aspect);

	std::vector<Sphere> sphere_data {
		{{-2,     1, 0},    1},
		{{ 0,     1, 0},    1},
		{{ 2,     1, 0},    1},
		{{ 0, -1000, 0}, 1000}
	};
	sphere_list = std::make_unique<HittableList<Sphere>>(sphere_data);

	renderer = std::make_unique<Renderer>(render_width, render_height, cam, sphere_list.get());

	CUDA_ASSERT(cudaMallocHost(&host_output_framebuffer, sizeof(glm::vec4) * render_width * render_height));
}
FirstApp::~FirstApp() {
	CUDA_ASSERT(cudaFreeHost(host_output_framebuffer));
}

void write_renderbuffer_png(std::string filepath, uint32_t width, uint32_t height, glm::vec4* data);
void FirstApp::Run() {
	renderer->Render();
	renderer->DownloadRenderbuffer(host_output_framebuffer);
	write_renderbuffer_png("../renders/test_010.png"s, render_width, render_height, host_output_framebuffer);
}

void write_renderbuffer_png(std::string filepath, uint32_t width, uint32_t height, glm::vec4* data) {
	uint8_t* output_image_data = new uint8_t[width * height * 4];
	for (int i = 0; i < width * height; i++) {
		output_image_data[i * 4 + 0] = static_cast<uint8_t>(data[i][0] * 255.999f);
		output_image_data[i * 4 + 1] = static_cast<uint8_t>(data[i][1] * 255.999f);
		output_image_data[i * 4 + 2] = static_cast<uint8_t>(data[i][2] * 255.999f);
		output_image_data[i * 4 + 3] = static_cast<uint8_t>(data[i][3] * 255.999f);
	}

	stbi_flip_vertically_on_write(true);
	stbi_write_png(filepath.c_str(), width, height, 4, output_image_data, sizeof(uint8_t) * width * 4);
	delete[] output_image_data;
}