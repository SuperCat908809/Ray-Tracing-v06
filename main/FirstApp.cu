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

#include "cu_Cameras.cuh"
#include "Renderer.cuh"

#include "cuHostRND.h"

template <typename T, typename... Args> requires std::constructible_from<T, Args...>
__global__ void makeOnDevice(T* dst_T, Args... args) {
	if (threadIdx.x != 0 || blockIdx.x != 0) return;

	new (dst_T) T(args...);
}

class SceneBook1FinaleFactory {

	void _make_sphere(int a, int b) {
	#define rnd (host_rnd.next())


		/*
		float choose_mat = RND;
		vec3 center(a+RND,0.2,b+RND);
		if(choose_mat < 0.8f) {
			d_list[i++] = new sphere(center, 0.2,
										new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
		}
		else if(choose_mat < 0.95f) {
			d_list[i++] = new sphere(center, 0.2,
										new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
		}
		else {
			d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
		}
		*/

		float choose_mat = rnd;
		glm::vec3 center(a + rnd, 0.2f, b + rnd);
		if (choose_mat < 0.8f) {
			auto mat = dobj<LambertianAbstract>::Make(glm::vec3(rnd * rnd, rnd * rnd, rnd * rnd));
			glm::vec3 center1 = center + glm::vec3(0, rnd * 0.5f, 0);
			auto sp = dobj<MovingSphereHittable>::Make(center, center1, 0.2f, mat.getPtr());
			materials.push_back(std::move(mat));
			sphere_list.push_back(std::move(sp));

			glm::vec3 r(0.2f);
			aabb sphere_bounds0(center - r, center + r);
			aabb sphere_bounds1(center1 - r, center1 + r);
			bounds += aabb(sphere_bounds0, sphere_bounds1);
		}
		else if (choose_mat < 0.95f) {
			auto mat = dobj<MetalAbstract>::Make(glm::vec3(0.5f * (1.0f + rnd), 0.5f * (1.0f + rnd), 0.5f * (1.0f * rnd)), 0.5f * rnd);
			auto sp = dobj<SphereHittable>::Make(center, 0.2f, mat.getPtr());
			materials.push_back(std::move(mat));
			sphere_list.push_back(std::move(sp));

			bounds += aabb(center - 2.0f, center + 2.0f);
		}
		else {
			auto mat = dobj<DielectricAbstract>::Make(glm::vec3(1.0f), 1.5f);
			auto sp = dobj<SphereHittable>::Make(center, 0.2f, mat.getPtr());
			materials.push_back(std::move(mat));
			sphere_list.push_back(std::move(sp));

			bounds += aabb(center - 2.0f, center + 2.0f);
		}
	}

	void _populate_world() {
		// ground sphere
		auto ground_mat = dobj<LambertianAbstract>::Make(glm::vec3(0.5f));
		//sphere_params.push_back({ glm::vec3(0,-1000,0), 1000, ground_mat.getPtr() });
		auto ground_sphere = dobj<SphereHittable>::Make(glm::vec3(0, -1000, -1), 1000, ground_mat.getPtr());
		materials.push_back(std::move(ground_mat));
		sphere_list.push_back(std::move(ground_sphere));

		bounds += aabb(glm::vec3(0, -1000, -1) - 1000.0f, glm::vec3(0, -1000, -1) + 1000.0f);


		for (int a = -11; a < 11; a++) {
			for (int b = -11; b < 11; b++) {
				_make_sphere(a, b);
			}
		}

		auto center_mat = dobj<DielectricAbstract>::Make(glm::vec3(1.0f), 1.5f);
		//sphere_params.push_back({ glm::vec3(0,1,0),1, center_mat.getPtr()});
		auto center_sphere = dobj<SphereHittable>::Make(glm::vec3(0, 1, 0), 1, center_mat.getPtr());
		materials.push_back(std::move(center_mat));
		sphere_list.push_back(std::move(center_sphere));

		bounds += aabb(glm::vec3(0, 1, 0) - 1.0f, glm::vec3(0, 1, 0) + 1.0f);


		auto left_mat = dobj<LambertianAbstract>::Make(glm::vec3(0.4f, 0.2f, 0.1f));
		//sphere_params.push_back({ glm::vec3(-4,1,0),1,left_mat.getPtr()});
		auto left_sphere = dobj<SphereHittable>::Make(glm::vec3(-4, 1, 0), 1, left_mat.getPtr());
		materials.push_back(std::move(left_mat));
		sphere_list.push_back(std::move(left_sphere));

		bounds += aabb(glm::vec3(-4, 1, 0) - 1.0f, glm::vec3(-4, 1, 0) + 1.0f);



		auto right_mat = dobj<MetalAbstract>::Make(glm::vec3(0.7f, 0.6f, 0.5f), 0);
		//sphere_params.push_back({ glm::vec3(4,1,0),1,right_mat.getPtr()});
		auto right_sphere = dobj<SphereHittable>::Make(glm::vec3(4, 1, 0), 1, right_mat.getPtr());
		materials.push_back(std::move(right_mat));
		sphere_list.push_back(std::move(right_sphere));

		bounds += aabb(glm::vec3(4, 1, 0) - 1.0f, glm::vec3(4, 1, 0) + 1.0f);
	}

	

	std::vector<Material*> materials;
	std::vector<Hittable*> sphere_list;
	cuHostRND host_rnd{ 512, 1984 };
	aabb bounds{};

public:

	static _SceneDescription MakeScene() {
		SceneBook1FinaleFactory factory{};

		factory._populate_world();
		darray<Hittable*> sphere_list = makePtrArray(factory.sphere_list);		
		auto world_list = dobj<HittableList>::Make(sphere_list.getPtr(), sphere_list.getLength(), factory.bounds);

		return _SceneDescription {
			std::move(factory.materials),
			std::move(factory.sphere_list),
			std::move(sphere_list),
			std::move(world_list)
		};
	}
};


class SceneBook2BVHNodeFactory {

	void push_sphere(dobj<Hittable>&& sp, const aabb& b) {
		bvh_node_data.push_back(std::make_tuple(sp.getPtr(), b));
		sphere_list.push_back(std::forward<dobj<Hittable>>(sp));
	}

	void _make_sphere(int a, int b) {
#define rnd (host_rnd.next())


		/*
		float choose_mat = RND;
		vec3 center(a+RND,0.2,b+RND);
		if(choose_mat < 0.8f) {
			d_list[i++] = new sphere(center, 0.2,
										new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
		}
		else if(choose_mat < 0.95f) {
			d_list[i++] = new sphere(center, 0.2,
										new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
		}
		else {
			d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
		}
		*/

		float choose_mat = rnd;
		glm::vec3 center(a + rnd, 0.2f, b + rnd);
		if (choose_mat < 0.8f) {
			auto mat = dobj<LambertianAbstract>::Make(glm::vec3(rnd * rnd, rnd * rnd, rnd * rnd));
			glm::vec3 center1 = center + glm::vec3(0, rnd * 0.5f, 0);
			auto sp = dobj<MovingSphereHittable>::Make(center, center1, 0.2f, mat.getPtr());
			material_list.push_back(std::move(mat));
			//sphere_list.push_back(std::move(sp));

			glm::vec3 r(0.2f);
			aabb sphere_bounds0(center - r, center + r);
			aabb sphere_bounds1(center1 - r, center1 + r);
			push_sphere(std::move(sp), aabb(sphere_bounds0, sphere_bounds1));
		}
		else if (choose_mat < 0.95f) {
			auto mat = dobj<MetalAbstract>::Make(glm::vec3(0.5f * (1.0f + rnd), 0.5f * (1.0f + rnd), 0.5f * (1.0f * rnd)), 0.5f * rnd);
			auto sp = dobj<SphereHittable>::Make(center, 0.2f, mat.getPtr());
			material_list.push_back(std::move(mat));
			//sphere_list.push_back(std::move(sp));

			push_sphere(std::move(sp), aabb(center - 2.0f, center + 2.0f));
		}
		else {
			auto mat = dobj<DielectricAbstract>::Make(glm::vec3(1.0f), 1.5f);
			auto sp = dobj<SphereHittable>::Make(center, 0.2f, mat.getPtr());
			material_list.push_back(std::move(mat));
			//sphere_list.push_back(std::move(sp));

			push_sphere(std::move(sp), aabb(center - 2.0f, center + 2.0f));
		}
	}

	void _populate_world() {
		// ground sphere
		auto ground_mat = dobj<LambertianAbstract>::Make(glm::vec3(0.5f));
		//sphere_params.push_back({ glm::vec3(0,-1000,0), 1000, ground_mat.getPtr() });
		auto ground_sphere = dobj<SphereHittable>::Make(glm::vec3(0, -1000, -1), 1000, ground_mat.getPtr());
		material_list.push_back(std::move(ground_mat));
		//sphere_list.push_back(std::move(ground_sphere));

		push_sphere(std::move(ground_sphere), aabb(glm::vec3(0, -1000, -1) - 1000.0f, glm::vec3(0, -1000, -1) + 1000.0f));


		for (int a = -11; a < 11; a++) {
			for (int b = -11; b < 11; b++) {
				_make_sphere(a, b);
			}
		}

		auto center_mat = dobj<DielectricAbstract>::Make(glm::vec3(1.0f), 1.5f);
		//sphere_params.push_back({ glm::vec3(0,1,0),1, center_mat.getPtr()});
		auto center_sphere = dobj<SphereHittable>::Make(glm::vec3(0, 1, 0), 1, center_mat.getPtr());
		material_list.push_back(std::move(center_mat));
		//sphere_list.push_back(std::move(center_sphere));

		push_sphere(std::move(center_sphere), aabb(glm::vec3(0, 1, 0) - 1.0f, glm::vec3(0, 1, 0) + 1.0f));


		auto left_mat = dobj<LambertianAbstract>::Make(glm::vec3(0.4f, 0.2f, 0.1f));
		//sphere_params.push_back({ glm::vec3(-4,1,0),1,left_mat.getPtr()});
		auto left_sphere = dobj<SphereHittable>::Make(glm::vec3(-4, 1, 0), 1, left_mat.getPtr());
		material_list.push_back(std::move(left_mat));
		//sphere_list.push_back(std::move(left_sphere));

		push_sphere(std::move(left_sphere), aabb(glm::vec3(-4, 1, 0) - 1.0f, glm::vec3(-4, 1, 0) + 1.0f));



		auto right_mat = dobj<MetalAbstract>::Make(glm::vec3(0.7f, 0.6f, 0.5f), 0);
		//sphere_params.push_back({ glm::vec3(4,1,0),1,right_mat.getPtr()});
		auto right_sphere = dobj<SphereHittable>::Make(glm::vec3(4, 1, 0), 1, right_mat.getPtr());
		material_list.push_back(std::move(right_mat));
		//sphere_list.push_back(std::move(right_sphere));

		push_sphere(std::move(right_sphere), aabb(glm::vec3(4, 1, 0) - 1.0f, glm::vec3(4, 1, 0) + 1.0f));
	}

	aabb build_bvh() { srand(0); return std::get<1>(_build_bvh(0, bvh_node_data.size())); }
	std::tuple<Hittable*, aabb> _build_bvh(size_t start, size_t end) {
		int axis = rand() % 3;
		auto comparitor = (axis == 0) ? box_x_compare
						: ((axis == 1) ? box_y_compare
						:				box_z_compare);

		size_t object_span = end - start;

		if (object_span == 1) {
			return bvh_node_data[start];
		}
		else if (object_span == 2) {
		#if 0
			Hittable* left;
			Hittable* right;

			if (comparitor(bounds_list[start], bounds_list[start + 1])) {
				left = sphere_list[start].getPtr();
				right = sphere_list[start + 1].getPtr();
			}
			else {
				left = sphere_list[start + 1].getPtr();
				right = sphere_list[start].getPtr();
			}
		#endif	

			Hittable* left = std::get<0>(bvh_node_data[start]);
			Hittable* right = std::get<0>(bvh_node_data[start + 1]);
			aabb node_bounds = aabb(std::get<1>(bvh_node_data[start]), std::get<1>(bvh_node_data[start + 1]));

			auto node = dobj<bvh_node>::Make(left, right, node_bounds);
			Hittable* ret_ptr = node.getPtr();
			sphere_list.push_back(std::move(node));
			return std::make_tuple(ret_ptr, node_bounds);
		}
		else {
			std::sort(bvh_node_data.begin() + start, bvh_node_data.begin() + end, [&, comparitor](const auto& a, const auto& b) -> bool {
				return comparitor(std::get<1>(a), std::get<1>(b));
				});

			//std::sort()

			auto mid = (start + end) / 2;
			std::tuple<Hittable*, aabb> left = _build_bvh(start, mid);
			std::tuple<Hittable*, aabb> right = _build_bvh(mid, end);

			aabb node_bounds = aabb(std::get<1>(left), std::get<1>(right));
			auto node = dobj<bvh_node>::Make(std::get<0>(left), std::get<0>(right), node_bounds);
			Hittable* ret_ptr = node.getPtr();
			sphere_list.push_back(std::move(node));
			return std::make_tuple(ret_ptr, node_bounds);
		}
	}

	std::vector<dobj<Material>> material_list;
	std::vector<dobj<Hittable>> sphere_list;
	std::vector<std::tuple<Hittable*, aabb>> bvh_node_data;
	cuHostRND host_rnd{ 512, 1984 };

public:

	static _SceneDescription MakeScene() {
		SceneBook2BVHNodeFactory factory{};

		factory._populate_world();
		int n = factory.sphere_list.size();
		aabb scene_bounds = factory.build_bvh();
		darray<Hittable*> sphere_list = makePtrArray(factory.sphere_list.data(), n);
		auto world_list = dobj<HittableList>::Make(sphere_list.getPtr(), n, scene_bounds);
		//auto world_list = dobj<HittableList>::Make(sphere_list[factory.sphere_list.size() - 1], 1ull, factory.sphere_list[factory.sphere_list.size() - 1]);
		//auto world_list = dobj<HittableList>::Make(sphere_list.getPtr(), sphere_list.getLength(), factory.bounds);

		return _SceneDescription{
			std::move(factory.material_list),
			std::move(factory.sphere_list),
			std::move(sphere_list),
			std::move(world_list)
		};
	}
};


FirstApp FirstApp::MakeApp() {
	uint32_t _width = 1280;
	uint32_t _height = 720;

	glm::vec3 lookfrom(13, 2, 3);
	glm::vec3 lookat(0, 0, 0);
	glm::vec3 up(0, 1, 0);
	float fov = 30.0f;
	float aspect = _width / (float)_height;
	MotionBlurCamera cam(lookfrom, lookat, up, fov, aspect, 0.1f, 1.0f);

	_SceneDescription scene_desc = SceneBook1FinaleFactory::MakeScene();

	Renderer renderer = Renderer::MakeRenderer(_width, _height, 8, 4, cam, scene_desc.world_list.getPtr());

	glm::vec4* host_output_framebuffer{};
	CUDA_ASSERT(cudaMallocHost(&host_output_framebuffer, sizeof(glm::vec4) * _width * _height));

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
	CUDA_ASSERT(cudaFreeHost(m.host_output_framebuffer));
}

void write_renderbuffer_png(std::string filepath, uint32_t width, uint32_t height, glm::vec4* data);
void FirstApp::Run() {
	m.renderer.Render();
	m.renderer.DownloadRenderbuffer(m.host_output_framebuffer);
	write_renderbuffer_png("../renders/Book 2/test_003.png"s, m.render_width, m.render_height, m.host_output_framebuffer);
}

void write_renderbuffer_png(std::string filepath, uint32_t width, uint32_t height, glm::vec4* data) {
	//uint8_t* output_image_data = new uint8_t[width * height * 4];
	std::vector<uint8_t> output_image_data;
	output_image_data.reserve(width * height * 4);
	for (uint32_t i = 0; i < width * height; i++) {
		output_image_data.push_back(static_cast<uint8_t>(data[i][0] * 255.999f));
		output_image_data.push_back(static_cast<uint8_t>(data[i][1] * 255.999f));
		output_image_data.push_back(static_cast<uint8_t>(data[i][2] * 255.999f));
		output_image_data.push_back(static_cast<uint8_t>(data[i][3] * 255.999f));
	}

	stbi_flip_vertically_on_write(true);
	stbi_write_png(filepath.c_str(), width, height, 4, output_image_data.data(), sizeof(uint8_t) * width * 4);
	//delete[] output_image_data;
}