#ifndef FIRST_APP_CLASS_H
#define FIRST_APP_CLASS_H

#define _USE_MATH_DEFINES
#include <math.h>
#include <string>
using namespace std::string_literals;
#include <assert.h>
#include <stdexcept>
#include <iostream>
#include <memory>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <glm/glm.hpp>
#include <stb/stb_image_write.h>

#include "cu_rtCommon.cuh"
#include "cu_Geometry.cuh"
#include "cu_Cameras.cuh"

#include "Renderer.cuh"

class FirstApp {

	uint32_t render_width{}, render_height{};
	PinholeCamera cam{};
	std::unique_ptr<Renderer> renderer{};
	std::unique_ptr<SphereList> sphere_list{};
	glm::vec4* host_output_framebuffer{};

public:

	FirstApp();
	~FirstApp();

	void Run();
};

#endif // FIRST_APP_CLASS_H //