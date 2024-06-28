#ifndef FIRST_APP_CLASS_H
#define FIRST_APP_CLASS_H

#include <inttypes.h>
#include <glm/glm.hpp>

#include "rt_engine/shaders/cu_Cameras.cuh"
#include "Renderer.cuh"
#include "rt_engine/geometry/Scenes.h"


class FirstApp {

	struct M {
		uint32_t render_width{}, render_height{};
		MotionBlurCamera cam{};
		glm::vec4* host_output_framebuffer{};
		Renderer renderer;

		SceneBook2BVH _sceneDesc;
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