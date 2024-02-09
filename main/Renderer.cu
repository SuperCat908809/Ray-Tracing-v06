#include "Renderer.cuh"
#include <device_launch_parameters.h>


__global__ void init_random_states(uint32_t width, uint32_t height, int seed, curandState_t* random_states) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= width || y >= height) return;
	int gid = y * width + x;
	curand_init(seed, gid, 0, &random_states[gid]);
}

Renderer::Renderer(uint32_t render_width, uint32_t render_height, uint32_t samples_per_pixel,
	const PinholeCamera& cam,
	const HittableList<Sphere>* h_sphere_list)
	: render_width(render_width), render_height(render_height), samples_per_pixel(samples_per_pixel), cam(cam) {

	CUDA_ASSERT(cudaMalloc(&d_sphere_list, sizeof(HittableList<Sphere>)));
	CUDA_ASSERT(cudaMemcpy(d_sphere_list, h_sphere_list, sizeof(HittableList<Sphere>), cudaMemcpyHostToDevice));

	CUDA_ASSERT(cudaMalloc(&d_output_buffer, sizeof(glm::vec4) * render_width * render_height));
	CUDA_ASSERT(cudaMalloc(&d_random_states, sizeof(curandState_t) * render_width * render_height));

	dim3 threads(8, 8, 1);
	dim3 blocks(ceilDiv(render_width, threads.x), ceilDiv(render_height, threads.y), 1);
	init_random_states<<<blocks, threads>>>(render_width, render_height, 1984, d_random_states);
	CUDA_ASSERT(cudaPeekAtLastError());
	CUDA_ASSERT(cudaDeviceSynchronize());
	CUDA_ASSERT(cudaGetLastError());
}
Renderer::~Renderer() {
	CUDA_ASSERT(cudaFree(d_sphere_list));
	CUDA_ASSERT(cudaFree(d_output_buffer));
	CUDA_ASSERT(cudaFree(d_random_states));
}

void Renderer::DownloadRenderbuffer(glm::vec4* host_dst) const {
	CUDA_ASSERT(cudaMemcpy(host_dst, d_output_buffer, sizeof(glm::vec4) * render_width * render_height, cudaMemcpyDeviceToHost));
}


struct LaunchParams {
	uint32_t render_width{};
	uint32_t render_height{};
	uint32_t samples_per_pixel{};
	PinholeCamera cam{};
	HittableList<Sphere>* sphere_list{};
	glm::vec4* output_buffer{};
	curandState_t* random_states{};
};
__global__ void kernel(LaunchParams p);

void Renderer::Render() {
	LaunchParams params{};
	params.render_width = render_width;
	params.render_height = render_height;
	params.samples_per_pixel = samples_per_pixel;
	params.cam = cam;
	params.sphere_list = d_sphere_list;
	params.output_buffer = d_output_buffer;
	params.random_states = d_random_states;

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

	glm::vec2 pixel_size = 1.0f / glm::vec2(p.render_width, p.render_height);

	int gid = y * p.render_width + x;
	curandState_t* random_state = &p.random_states[gid];
	// sample at a random position inside a circle of radius 0.3 pixels centered at the pixel
	glm::vec2 ndc = (glm::vec2(x, y) + glm::vec2(0.5f)) * pixel_size * 2.0f - 1.0f;
	glm::vec4 output_color{};

	for (int i = 0; i < p.samples_per_pixel; i++) {
		glm::vec2 rnd_ndc_sample = ndc + glm::cu_random_in_unit<2>(random_state) * pixel_size;
		Ray ray = p.cam.sample_ray(rnd_ndc_sample.x, rnd_ndc_sample.y);

		TraceRecord rec{};
		if (p.sphere_list->ClosestIntersection(ray, rec)) {
			output_color += glm::vec4(rec.n * 0.5f + 0.5f, 1.0f);
		}
		else {
			float t = glm::normalize(ray.d).y * 0.5f + 0.5f;
			output_color += (1 - t) * glm::vec4(0.1f, 0.2f, 0.4f, 1.0f) + t * glm::vec4(0.9f, 0.9f, 0.99f, 1.0f);
		}
	}

	p.output_buffer[gid] = output_color / (float)p.samples_per_pixel;
}