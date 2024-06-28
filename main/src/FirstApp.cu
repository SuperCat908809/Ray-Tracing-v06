#include "FirstApp.cuh"

#include <string>
#include <iostream>
#include <inttypes.h>
#include <glm/glm.hpp>
#include <stb/stb_image_write.h>

using namespace std::string_literals;

#include "rt_engine/shaders/cu_Cameras.cuh"
#include "Renderer.cuh"
#include "rt_engine/geometry/Scenes.h"


FirstApp FirstApp::MakeApp() {
	uint32_t _width = 1920 * 2;
	uint32_t _height = 1080 * 2;

	printf("Building MotionBlurCamera object... ");
	glm::vec3 lookfrom(13, 2, 3);
	glm::vec3 lookat(0, 0, 0);
	glm::vec3 up(0, 1, 0);
	float fov = 30.0f;
	float aspect = _width / (float)_height;
	MotionBlurCamera cam(lookfrom, lookat, up, fov, aspect, 0.1f, 1.0f);
	printf("done.\n");

	printf("Building SceneBook2BVH object...\n");
	SceneBook2BVH::Factory scene_factory{};
	SceneBook2BVH scene_desc = scene_factory.MakeScene();
	printf("SceneBook2BVH object built.\n");
		
	printf("Making Renderer object...\n");
	Renderer renderer = Renderer::MakeRenderer(_width, _height, 1024, 16, cam, scene_desc.getWorldPtr());
	printf("Renderer object built.\n");

	glm::vec4* host_output_framebuffer{};
	printf("Allocating host framebuffer... ");
	CUDA_ASSERT(cudaMallocHost(&host_output_framebuffer, sizeof(glm::vec4) * _width * _height));
	printf("done.\n");

	return FirstApp(M{
		_width,
		_height,
		cam,
		host_output_framebuffer,
		std::move(renderer),
		std::move(scene_desc),
	});
}
FirstApp::~FirstApp() {
	printf("Freeing host framebuffer allocation... ");
	CUDA_ASSERT(cudaFreeHost(m.host_output_framebuffer));
	printf("done.\n");
}

void write_renderbuffer(std::string filepath, uint32_t width, uint32_t height, glm::vec4* data);
void FirstApp::Run() {
	printf("Rendering scene...\n");
	m.renderer.Render();
	printf("Scene rendered.\n");

	printf("Downloading render to host framebuffer... ");
	m.renderer.DownloadRenderbuffer(m.host_output_framebuffer);
	printf("done.\n");

	printf("Writing render to disk... ");
	write_renderbuffer("../renders/Book 2/test_018.jpg"s, m.render_width, m.render_height, m.host_output_framebuffer);
	printf("done.\n");
}

void write_renderbuffer(std::string filepath, uint32_t width, uint32_t height, glm::vec4* data) {
	//uint8_t* output_image_data = new uint8_t[width * height * 4];
	std::vector<uint8_t> output_image_data;
	output_image_data.reserve(width * height * 3);
	for (uint32_t i = 0; i < width * height; i++) {
		output_image_data.push_back(static_cast<uint8_t>(data[i][0] * 255.999f));
		output_image_data.push_back(static_cast<uint8_t>(data[i][1] * 255.999f));
		output_image_data.push_back(static_cast<uint8_t>(data[i][2] * 255.999f));
		//output_image_data.push_back(static_cast<uint8_t>(data[i][3] * 255.999f));
	}

	stbi_flip_vertically_on_write(true);
	stbi_write_jpg(filepath.c_str(), width, height, 3, output_image_data.data(), 95);
	//delete[] output_image_data;
}