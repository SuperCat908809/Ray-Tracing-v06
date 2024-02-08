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
#include "cu_geometry.cuh"
#include "cu_Cameras.cuh"

struct LaunchParams {
	uint32_t render_width{};
	uint32_t render_height{};
	PinholeCamera cam{};
	Sphere sphere{};
	glm::vec4* output_buffer{};
};

__global__ void kernel(LaunchParams p) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= p.render_width || y >= p.render_height) return;

	int gid = y * p.render_width + x;
	float u = x / (p.render_width - 1.0f) * 2 - 1;
	float v = y / (p.render_height - 1.0f) * 2 - 1;
	glm::vec2 ndc(u, v);

	Ray ray = p.cam.sample_ray(u, v);

	TraceRecord rec{};
	glm::vec4 output_color{};
	if (p.sphere.Trace(ray, rec)) {
		output_color = glm::vec4(rec.n * 0.5f + 0.5f, 1.0f);
	}
	else {
		float t = glm::normalize(ray.d).y * 0.5f + 0.5f;
		output_color = (1 - t) * glm::vec4(0.1f, 0.2f, 0.4f, 1.0f) + t * glm::vec4(0.9f, 0.9f, 0.99f, 1.0f);
	}

	p.output_buffer[gid] = output_color;
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

int main() {

	LaunchParams p{};
	p.render_width = 1280;
	p.render_height = 720;

	glm::vec3 lookfrom(0, 0, 4);
	glm::vec3 lookat(0, 0, 0);
	glm::vec3 up(0, 1, 0);
	float fov = 90.0f;
	float aspect = p.render_width / (float)p.render_height;
	p.cam = PinholeCamera(lookfrom, lookat, up, fov, aspect);

	Sphere sphere{};
	sphere.origin = glm::vec3(0, 0, 0);
	sphere.radius = 1;
	p.sphere = sphere;

	glm::vec4* h_framebuffer{};
	glm::vec4* d_framebuffer{};

	CUDA_ASSERT(cudaMallocHost(&h_framebuffer, sizeof(glm::vec4) * p.render_width * p.render_height));
	CUDA_ASSERT(cudaMalloc(&d_framebuffer, sizeof(glm::vec4) * p.render_width * p.render_height));

	p.output_buffer = d_framebuffer;

	dim3 threads{ 8, 8, 1 };
	dim3 blocks = dim3((p.render_width + threads.x - 1) / threads.x, (p.render_height + threads.y - 1) / threads.y, 1);
	kernel<<<blocks, threads>>>(p);
	CUDA_ASSERT(cudaPeekAtLastError());
	CUDA_ASSERT(cudaDeviceSynchronize());


	CUDA_ASSERT(cudaMemcpy(h_framebuffer, d_framebuffer, sizeof(glm::vec4) * p.render_width * p.render_height, cudaMemcpyDeviceToHost));

	write_renderbuffer_png("../renders/test_006.png"s, p.render_width, p.render_height, h_framebuffer);


	CUDA_ASSERT(cudaFreeHost(h_framebuffer));
	CUDA_ASSERT(cudaFree(d_framebuffer));

	CUDA_ASSERT(cudaDeviceReset());

	return 0;
}