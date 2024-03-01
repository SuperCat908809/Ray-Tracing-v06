#ifndef FIRST_APP_CLASS_H
#define FIRST_APP_CLASS_H

#include <string>
using namespace std::string_literals;
#include <iostream>

#include "cu_rtCommon.cuh"
#include "cu_Geometry.cuh"
#include "cu_Cameras.cuh"
#include "cu_Materials.cuh"

#include "handled_device_abstracts.cuh"
#include "Renderer.cuh"

struct _SceneDescription {
	dAbstractArray<Material> sphere_materials;
	dAbstractArray<Hittable> world_sphere_list;
	dAbstract<HittableList> world_list;
};

class FirstApp {

	struct M {
		uint32_t render_width{}, render_height{};
		PinholeCamera cam{};
		glm::vec4* host_output_framebuffer{};
		Renderer renderer;

		_SceneDescription _sceneDesc;
	} m;

	FirstApp(M m) : m(std::move(m)) {}

public:

	FirstApp(const FirstApp&) = delete;
	FirstApp& operator=(const FirstApp&) = delete;

	static FirstApp MakeApp();
	FirstApp(FirstApp&& other) : m(std::move(other.m)) {}
	~FirstApp();

	void Run();
};

#endif // FIRST_APP_CLASS_H //