#include "FirstApp.cuh"

#include <inttypes.h>
#include <string>
using namespace std::string_literals;
#include <iostream>
#include <vector>
#include <algorithm>
#include <tuple>

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <stb/stb_image_write.h>

#include "cuError.h"

#include "dobj.cuh"
#include "darray.cuh"
#include "dAbstracts.cuh"

#include "hittable.cuh"
#include "SphereHittable.cuh"
#include "HittableList.cuh"
#include "bvh_node.cuh"

#include "material.cuh"
#include "cu_Materials.cuh"

#include "texture.cuh"
#include "cu_Textures.cuh"

#include "cu_Cameras.cuh"
#include "Renderer.cuh"

#include "cuHostRND.h"


void SceneBook1::_delete() {
	CUDA_ASSERT(cudaFree(world));
	CUDA_ASSERT(cudaFree(hittable_list));
}

void SceneBook1::Factory::_populate_world() {
	
	Sphere ground_sphere = Sphere(glm::vec3(0, -1000, 0), 1000.0f);
	LambertianAbstract<Sphere>* ground_mat = newOnDevice<LambertianAbstract<Sphere>>(glm::vec3(0.5f));
	SphereHandle ground_handle = SphereHandle::MakeSphere(ground_sphere, ground_mat);
	world_bounds += ground_handle.getBounds();
	sphere_handles.push_back(std::move(ground_handle));

	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {

			#define rnd host_rnd.next()

			float choose_mat = rnd;
			glm::vec3 center(a + rnd, 0.2f, b + rnd);

			if (choose_mat < 0.8f) {
				auto material = newOnDevice<LambertianAbstract<MovingSphere>>(glm::vec3(rnd * rnd, rnd * rnd, rnd * rnd));
				glm::vec3 center1 = center + glm::vec3(0, rnd * 0.5f, 0);
				auto moving_sphere = MovingSphere(center, center1, 0.2f);
				auto handle = SphereHandle::MakeMovingSphere(moving_sphere, material);
				world_bounds += handle.getBounds();
				sphere_handles.push_back(std::move(handle));
			}
			else if (choose_mat < 0.95f) {
				auto material = newOnDevice<MetalAbstract<Sphere>>(glm::vec3(0.5f * (1.0f + rnd), 0.5f * (1.0f + rnd), 0.5f * (1.0f + rnd)), 0.5f * rnd);
				auto sphere = Sphere(center, 0.2f);
				auto handle = SphereHandle::MakeSphere(sphere, material);
				world_bounds += handle.getBounds();
				sphere_handles.push_back(std::move(handle));
			}
			else {
				auto material = newOnDevice<DielectricAbstract<Sphere>>(glm::vec3(1.0f), 1.5f);
				auto sphere = Sphere(center, 0.2f);
				auto handle = SphereHandle::MakeSphere(sphere, material);
				world_bounds += handle.getBounds();
				sphere_handles.push_back(std::move(handle));
			}
		}
	}

	Sphere center_sphere = Sphere(glm::vec3(0, 1, 0), 1);
	DielectricAbstract<Sphere>* center_mat = newOnDevice<DielectricAbstract<Sphere>>(glm::vec3(1.0f), 1.5f);
	auto center_handle = SphereHandle::MakeSphere(center_sphere, center_mat);
	world_bounds += center_handle.getBounds();
	sphere_handles.push_back(std::move(center_handle));

	Sphere left_sphere = Sphere(glm::vec3(-4, 1, 0), 1);
	LambertianAbstract<Sphere>* left_mat = newOnDevice<LambertianAbstract<Sphere>>(glm::vec3(0.4f, 0.2f, 0.1f));
	auto left_handle = SphereHandle::MakeSphere(left_sphere, left_mat);
	world_bounds += left_handle.getBounds();
	sphere_handles.push_back(std::move(left_handle));

	Sphere right_sphere = Sphere(glm::vec3(4, 1, 0), 1);
	MetalAbstract<Sphere>* right_mat = newOnDevice<MetalAbstract<Sphere>>(glm::vec3(0.7f, 0.6f, 0.5f), 0);
	auto right_handle = SphereHandle::MakeSphere(right_sphere, right_mat);
	world_bounds += right_handle.getBounds();
	sphere_handles.push_back(std::move(right_handle));
}

SceneBook1 SceneBook1::Factory::MakeScene() {

	printf("Populating world... ");
	_populate_world();
	printf("done.\n");

	printf("Building world's HittableList... ");
	std::vector<const Hittable*> hittable_vec{};
	for (int i = 0; i < sphere_handles.size(); i++) {
		const Hittable* ptr = sphere_handles[i].getHittablePtr();
		hittable_vec.push_back(ptr);
	}

	CUDA_ASSERT(cudaMalloc((void**)&hittable_list, sizeof(Hittable*) * hittable_vec.size()));
	CUDA_ASSERT(cudaMemcpy(hittable_list, hittable_vec.data(), sizeof(Hittable*) * hittable_vec.size(), cudaMemcpyHostToDevice));
	world = newOnDevice<HittableList>(
		const_cast<const Hittable**>(hittable_list),
		(int)hittable_vec.size(),
		world_bounds
	);
	printf("done.\n");

	SceneBook1 scene;

	scene.world_bounds = world_bounds;
	scene.world = world;
	scene.hittable_list = hittable_list;
	scene.sphere_handles = std::move(sphere_handles);

	return scene;
}

SceneBook1::SceneBook1(SceneBook1&& scene) {
	world_bounds = scene.world_bounds;
	
	world = scene.world;
	scene.world = nullptr;

	hittable_list = scene.hittable_list;
	scene.hittable_list = nullptr;

	sphere_handles = std::move(scene.sphere_handles);
}

SceneBook1& SceneBook1::operator=(SceneBook1&& scene) {
	_delete();

	world_bounds = scene.world_bounds;

	world = scene.world;
	scene.world = nullptr;

	hittable_list = scene.hittable_list;
	scene.hittable_list = nullptr;

	sphere_handles = std::move(scene.sphere_handles);

	return *this;
}

SceneBook1::~SceneBook1() {
	_delete();
}


FirstApp FirstApp::MakeApp() {
	uint32_t _width = 1280;
	uint32_t _height = 720;

	printf("Building MotionBlurCamera object... ");
	glm::vec3 lookfrom(13, 2, 3);
	glm::vec3 lookat(0, 0, 0);
	glm::vec3 up(0, 1, 0);
	float fov = 30.0f;
	float aspect = _width / (float)_height;
	MotionBlurCamera cam(lookfrom, lookat, up, fov, aspect, 0.1f, 1.0f);
	printf("done.\n");

	printf("Building SceneBook1 object...\n");
	SceneBook1::Factory scene_factory{};
	SceneBook1 scene_desc = scene_factory.MakeScene();
	printf("SceneBook1 object built.\n");
		
	printf("Making Renderer object...\n");
	Renderer renderer = Renderer::MakeRenderer(_width, _height, 8, 12, cam, scene_desc.getWorldPtr());
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
	write_renderbuffer("../renders/Book 2/test_011.jpg"s, m.render_width, m.render_height, m.host_output_framebuffer);
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