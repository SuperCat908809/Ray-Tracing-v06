#include "Renderer.cuh"
#include "cu_Materials.cuh"
#include <device_launch_parameters.h>


__global__ void init_random_states(uint32_t width, uint32_t height, int seed, curandState_t* random_states) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= width || y >= height) return;
	int gid = y * width + x;
	curand_init(seed, gid, 0, &random_states[gid]);
}

Renderer::Renderer(uint32_t render_width, uint32_t render_height,
	uint32_t samples_per_pixel, uint32_t max_depth,
	const PinholeCamera& cam,
	const HittableList<Sphere>* h_sphere_list)
	: render_width(render_width), render_height(render_height),
	samples_per_pixel(samples_per_pixel), max_depth(max_depth),
	cam(cam) {

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
	uint32_t max_depth{};
	PinholeCamera cam{};
	HittableList<Sphere>* sphere_list{};
	MetalMaterial mat{};
	glm::vec4* output_buffer{};
	curandState_t* random_states{};
};
__global__ void kernel(LaunchParams p);

void Renderer::Render() {
	LaunchParams params{};
	params.render_width = render_width;
	params.render_height = render_height;
	params.samples_per_pixel = samples_per_pixel;
	params.max_depth = max_depth;
	params.cam = cam;
	params.sphere_list = d_sphere_list;
	params.mat = MetalMaterial(glm::vec3(1.0f));
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
	glm::vec3 accumulated_radiance(0.0f);
	int samples_collected = 0;

	for (int sample_idx = 0; sample_idx < p.samples_per_pixel; sample_idx++) {
		glm::vec2 rnd_ndc_sample = ndc + glm::cu_random_in_unit_vec<2>(random_state) * pixel_size;
		Ray cur_ray = p.cam.sample_ray(rnd_ndc_sample.x, rnd_ndc_sample.y);
		glm::vec3 cur_attenuation(1.0f);

		for (int depth = 0; depth < p.max_depth; depth++) {
			TraceRecord rec{};

			if (p.sphere_list->ClosestIntersection(cur_ray, rec)) {
				Ray scatter_ray{};
				glm::vec3 attenuation{};

				if (p.mat.Scatter(cur_ray, rec, random_state, scatter_ray, attenuation)) {
					cur_ray = scatter_ray;
					cur_attenuation *= attenuation;
					continue;
				}
				break;
			}
			else {
				float t = glm::normalize(cur_ray.d).y * 0.5f + 0.5f;
				glm::vec3 sky_radiance = glm::linear_interpolate(glm::vec3(0.1f, 0.2f, 0.4f), glm::vec3(0.9f, 0.9f, 0.99f), t);
				accumulated_radiance += cur_attenuation * sky_radiance;
				samples_collected++;
				break;
			}
		}
	}

	glm::vec4 output_color = glm::vec4(accumulated_radiance / (float)samples_collected, 1.0f);

	output_color[0] = sqrtf(output_color[0]);
	output_color[1] = sqrtf(output_color[1]);
	output_color[2] = sqrtf(output_color[2]);

	p.output_buffer[gid] = output_color;
}