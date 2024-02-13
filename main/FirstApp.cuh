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

class FirstApp {

	uint32_t render_width{}, render_height{};
	PinholeCamera cam{};
	std::unique_ptr<Renderer> renderer{};
	glm::vec4* host_output_framebuffer{};

	// gpu memory
	std::unique_ptr<HandledDeviceAbstractArray<Hittable>> world_sphere_list{};
	std::unique_ptr<HandledDeviceAbstract<HittableList>> world_list{};
	std::unique_ptr<HandledDeviceAbstract<LambertianAbstract>> ground_mat{};
	std::unique_ptr<HandledDeviceAbstract<LambertianAbstract>> center_mat{};
	std::unique_ptr<HandledDeviceAbstract<DielectricAbstract>> left_mat{};
	std::unique_ptr<HandledDeviceAbstract<     MetalAbstract>> right_mat{};

public:

	FirstApp();
	~FirstApp();

	void Run();
};

#endif // FIRST_APP_CLASS_H //