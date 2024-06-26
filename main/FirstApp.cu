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
#include "timers.h"

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
#include "cuda_utils.cuh"


void SceneBook1::_delete() {
	CUDA_ASSERT(cudaFree(world));
	CUDA_ASSERT(cudaFree(hittable_list));

	world = nullptr;
	hittable_list = nullptr;
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
	hittable_list = scene.hittable_list;
	sphere_handles = std::move(scene.sphere_handles);

	scene.world = nullptr;
	scene.hittable_list = nullptr;
}

SceneBook1& SceneBook1::operator=(SceneBook1&& scene) {
	_delete();

	world_bounds = scene.world_bounds;
	world = scene.world;
	hittable_list = scene.hittable_list;
	sphere_handles = std::move(scene.sphere_handles);

	scene.world = nullptr;
	scene.hittable_list = nullptr;

	return *this;
}

SceneBook1::~SceneBook1() {
	_delete();
}



void SceneBook2BVH::_delete() {
	CUDA_ASSERT(cudaFree(world));
	CUDA_ASSERT(cudaFree(hittable_list));

#if 0
	for (int i = 0; i < bvh_nodes.size(); i++) {
		CUDA_ASSERT(cudaFree(bvh_nodes[i]));
	}
	bvh_nodes.resize(0);
#endif

	world = nullptr;
	hittable_list = nullptr;
}

void SceneBook2BVH::Factory::_populate_world() {

	Sphere ground_sphere = Sphere(glm::vec3(0, -1000, 0), 1000.0f);
	LambertianAbstract<Sphere>* ground_mat = newOnDevice<LambertianAbstract<Sphere>>(glm::vec3(0.5f));
	SphereHandle ground_handle = SphereHandle::MakeSphere(ground_sphere, ground_mat);
	//world_bounds += ground_handle.getBounds();
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
				//world_bounds += handle.getBounds();
				sphere_handles.push_back(std::move(handle));
			}
			else if (choose_mat < 0.95f) {
				auto material = newOnDevice<MetalAbstract<Sphere>>(glm::vec3(0.5f * (1.0f + rnd), 0.5f * (1.0f + rnd), 0.5f * (1.0f + rnd)), 0.5f * rnd);
				auto sphere = Sphere(center, 0.2f);
				auto handle = SphereHandle::MakeSphere(sphere, material);
				//world_bounds += handle.getBounds();
				sphere_handles.push_back(std::move(handle));
			}
			else {
				auto material = newOnDevice<DielectricAbstract<Sphere>>(glm::vec3(1.0f), 1.5f);
				auto sphere = Sphere(center, 0.2f);
				auto handle = SphereHandle::MakeSphere(sphere, material);
				//world_bounds += handle.getBounds();
				sphere_handles.push_back(std::move(handle));
			}
		}
	}

	Sphere center_sphere = Sphere(glm::vec3(0, 1, 0), 1);
	DielectricAbstract<Sphere>* center_mat = newOnDevice<DielectricAbstract<Sphere>>(glm::vec3(1.0f), 1.5f);
	auto center_handle = SphereHandle::MakeSphere(center_sphere, center_mat);
	//world_bounds += center_handle.getBounds();
	sphere_handles.push_back(std::move(center_handle));

	Sphere left_sphere = Sphere(glm::vec3(-4, 1, 0), 1);
	LambertianAbstract<Sphere>* left_mat = newOnDevice<LambertianAbstract<Sphere>>(glm::vec3(0.4f, 0.2f, 0.1f));
	auto left_handle = SphereHandle::MakeSphere(left_sphere, left_mat);
	//world_bounds += left_handle.getBounds();
	sphere_handles.push_back(std::move(left_handle));

	Sphere right_sphere = Sphere(glm::vec3(4, 1, 0), 1);
	MetalAbstract<Sphere>* right_mat = newOnDevice<MetalAbstract<Sphere>>(glm::vec3(0.7f, 0.6f, 0.5f), 0);
	auto right_handle = SphereHandle::MakeSphere(right_sphere, right_mat);
	//world_bounds += right_handle.getBounds();
	sphere_handles.push_back(std::move(right_handle));
}


#if 0
aabb _get_partition_bounds(std::vector<std::tuple<aabb, const Hittable*>>& arr, int start, int end) {
	aabb partition_bounds{};
	for (int i = start; i < end; i++) {
		partition_bounds += std::get<0>(arr[i]);
	}
	return partition_bounds;
}

const Hittable* SceneBook2BVH::Factory::_build_bvh_rec(
	std::vector<std::tuple<aabb, const Hittable*>>& arr, int start, int end) {
	aabb bounds = _get_partition_bounds(arr, start, end);
	int axis = bounds.longest_axis();
	auto comparator = (axis == 0) ? box_x_compare : ((axis == 1) ? box_y_compare : box_z_compare);

	int object_span = end - start;
	if (object_span == 1) {
		return std::get<1>(arr[start]);
	}
	else if (object_span == 2) {
		auto node_ptr = newOnDevice<bvh_node>(
			std::get<1>(arr[start + 0]),
			std::get<1>(arr[start + 1]),
			bounds
		);
		bvh_nodes.push_back(node_ptr);
		return node_ptr;
	}
	else {
		std::sort(arr.begin() + start, arr.begin() + end,
			[comparator](const std::tuple<aabb, const Hittable*>& a, const std::tuple<aabb, const Hittable*>& b) {
				return comparator(std::get<0>(a), std::get<0>(b));
			}
		);

		int mid = (start + end) / 2;
		auto left = _build_bvh_rec(arr, start, mid);
		auto right = _build_bvh_rec(arr, mid, end);
		auto node_ptr = newOnDevice<bvh_node>(left, right, bounds);
		bvh_nodes.push_back(node_ptr);
		return node_ptr;
	}
}

const Hittable* SceneBook2BVH::Factory::_build_bvh() {

	std::vector<std::tuple<aabb, const Hittable*>> objects;
	for (int i = 0; i < sphere_handles.size(); i++) {
		auto bounds = sphere_handles[i].getBounds();
		auto hittable_ptr = sphere_handles[i].getHittablePtr();

		objects.push_back({ bounds, hittable_ptr });
	}

	const Hittable* root_node = _build_bvh_rec(objects, 0, objects.size());
	return root_node;
}
#endif

SceneBook2BVH SceneBook2BVH::Factory::MakeScene() {

	printf("Populating world... ");
	hostTimer populate_timer{};
	populate_timer.start();

	_populate_world();

	populate_timer.end();
	printf("done in %fms.\n", populate_timer.elapsedms());


	printf("Building BVH... ");
	hostTimer bvh_timer{};
	bvh_timer.start();

	std::vector<std::tuple<aabb, const Hittable*>> objects;
	objects.reserve(sphere_handles.size());
	for (int i = 0; i < sphere_handles.size(); i++) {
		auto bounds = sphere_handles[i].getBounds();
		auto hittable_ptr = sphere_handles[i].getHittablePtr();

		objects.push_back({ bounds, hittable_ptr });
	}

	//const Hittable* root_node = _build_bvh();
	BVH_Handle::Factory bvh_factory(objects);
	BVH_Handle bvh_handle = bvh_factory.MakeBVH();
	const Hittable* bvh_ptr = bvh_handle.getBVHPtr();
	aabb world_bounds = bvh_handle.getBounds();

	bvh_timer.end();
	printf("done in %fms.\n", bvh_timer.elapsedms());


	printf("Building world's HittableList... ");
	//std::vector<const Hittable*> hittable_vec{};
	//for (int i = 0; i < sphere_handles.size(); i++) {
	//	const Hittable* ptr = sphere_handles[i].getHittablePtr();
	//	hittable_vec.push_back(ptr);
	//}

	Hittable** hittable_list;
	HittableList* world;

	CUDA_ASSERT(cudaMalloc((void**)&hittable_list, sizeof(Hittable**)));
	CUDA_ASSERT(cudaMemcpy(hittable_list, &bvh_ptr, sizeof(Hittable**), cudaMemcpyHostToDevice));
	world = newOnDevice<HittableList>(
		const_cast<const Hittable**>(hittable_list),
		1,
		world_bounds
	);
	printf("done.\n");

	SceneBook2BVH scene(std::move(bvh_handle));

	scene.world_bounds = world_bounds;
	scene.world = world;
	scene.hittable_list = hittable_list;
	scene.sphere_handles = std::move(sphere_handles);
	//scene.bvh_nodes = std::move(bvh_nodes);

	return scene;
}

SceneBook2BVH::SceneBook2BVH(SceneBook2BVH&& scene)
	: bvh(std::move(scene.bvh)) {
	world_bounds = scene.world_bounds;
	world = scene.world;
	hittable_list = scene.hittable_list;
	sphere_handles = std::move(scene.sphere_handles);
	//bvh_nodes = std::move(scene.bvh_nodes);

	scene.world = nullptr;
	scene.hittable_list = nullptr;
}

SceneBook2BVH& SceneBook2BVH::operator=(SceneBook2BVH&& scene) {
	_delete();

	world_bounds = scene.world_bounds;
	world = scene.world;
	hittable_list = scene.hittable_list;
	sphere_handles = std::move(scene.sphere_handles);
	//bvh_nodes = std::move(scene.bvh_nodes);
	bvh = std::move(scene.bvh);

	scene.world = nullptr;
	scene.hittable_list = nullptr;

	return *this;
}

SceneBook2BVH::~SceneBook2BVH() {
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

	printf("Building SceneBook2BVH object...\n");
	SceneBook2BVH::Factory scene_factory{};
	SceneBook2BVH scene_desc = scene_factory.MakeScene();
	printf("SceneBook2BVH object built.\n");
		
	printf("Making Renderer object...\n");
	Renderer renderer = Renderer::MakeRenderer(_width, _height, 32, 12, cam, scene_desc.getWorldPtr());
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
	write_renderbuffer("../renders/Book 2/test_012.jpg"s, m.render_width, m.render_height, m.host_output_framebuffer);
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