#define _USE_MATH_DEFINES
#include <math.h>
#include <string>
using namespace std::string_literals;
#include <assert.h>
#include <stdexcept>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <glm/glm.hpp>
#include <stb/stb_image_write.h>

#include "cu_rtCommon.cuh"
#include "cu_Geometry.cuh"
#include "cu_Cameras.cuh"

#include "Renderer.cuh"
#include "FirstApp.cuh"

int main() {

	{

		dAbstract<Material> mat_ptr2 = dAbstract<Material>::MakeAbstract<MetalAbstract>(glm::vec3(1.0f), 0.05f);

		FirstApp app = FirstApp::MakeApp();
		app.Run();
	}

	CUDA_ASSERT(cudaDeviceReset());

	return 0;
}