#include "Renderer.cuh"
#include <device_launch_parameters.h>


Renderer::Renderer(uint32_t render_width, uint32_t render_height,
	const PinholeCamera& cam,
	const SphereList* h_sphere_list)
	: render_width(render_width), render_height(render_height), cam(cam) {

	CUDA_ASSERT(cudaMalloc(&d_sphere_list, sizeof(SphereList)));
	CUDA_ASSERT(cudaMemcpy(d_sphere_list, h_sphere_list, sizeof(SphereList), cudaMemcpyHostToDevice));

	CUDA_ASSERT(cudaMalloc(&d_output_buffer, sizeof(glm::vec4) * render_width * render_height));
}
Renderer::~Renderer() {
	CUDA_ASSERT(cudaFree(d_sphere_list));
	CUDA_ASSERT(cudaFree(d_output_buffer));
}

void Renderer::DownloadRenderbuffer(glm::vec4* host_dst) const {
	CUDA_ASSERT(cudaMemcpy(host_dst, d_output_buffer, sizeof(glm::vec4) * render_width * render_height, cudaMemcpyDeviceToHost));
}


struct LaunchParams {
	uint32_t render_width{};
	uint32_t render_height{};
	PinholeCamera cam{};
	SphereList* sphere_list{};
	glm::vec4* output_buffer{};
};
__global__ void kernel(LaunchParams p);

void Renderer::Render() {
	LaunchParams params{};
	params.render_width = render_width;
	params.render_height = render_height;
	params.cam = cam;
	params.sphere_list = d_sphere_list;
	params.output_buffer = d_output_buffer;

	dim3 threads{ 8, 8, 1 };
	dim3 blocks = dim3((render_width + threads.x - 1) / threads.x, (render_height + threads.y - 1) / threads.y, 1);
	kernel<<<blocks, threads>>>(params);
	CUDA_ASSERT(cudaPeekAtLastError());
	CUDA_ASSERT(cudaDeviceSynchronize());
	CUDA_ASSERT(cudaGetLastError());
}

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
	if (p.sphere_list->ClosestIntersection(ray, rec)) {
		output_color = glm::vec4(rec.n * 0.5f + 0.5f, 1.0f);
	}
	else {
		float t = glm::normalize(ray.d).y * 0.5f + 0.5f;
		output_color = (1 - t) * glm::vec4(0.1f, 0.2f, 0.4f, 1.0f) + t * glm::vec4(0.9f, 0.9f, 0.99f, 1.0f);
	}

	p.output_buffer[gid] = output_color;
}