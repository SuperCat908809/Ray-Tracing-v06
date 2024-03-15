#include "Renderer.cuh"

#include <inttypes.h>
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include "cuError.h"

#include "darray.cuh"
#include "dobj.cuh"

#include "ray_data.cuh"

#include "cu_Cameras.cuh"
#include "material.cuh"
#include "hittable.cuh"

#include "cu_Materials.cuh"
#include "HittableList.cuh"

#include "cuThreadManagement.cuh"


__global__ void init_random_states(uint32_t width, uint32_t height, int seed, cuRandom* rngs) {
	int x = THREAD_GLOBAL_X_ID;
	int y = THREAD_GLOBAL_Y_ID;
	if (x >= width || y >= height) return;
	int gid = y * width + x;
	//curand_init(seed, gid, 0, &random_states[gid]);
	new (rngs + gid) cuRandom(seed, 0, gid);
}

void Renderer::_delete() {
	//if (m.d_output_buffer != nullptr)
	//	CUDA_ASSERT(cudaFree(m.d_output_buffer));
	//if (m.d_random_states != nullptr)
	//	CUDA_ASSERT(cudaFree(m.d_random_states));
}

Renderer Renderer::MakeRenderer(uint32_t render_width, uint32_t render_height,
	uint32_t samples_per_pixel, uint32_t max_depth,
	const PinholeCamera& cam,
	HittableList* d_world_ptr) {

	//glm::vec4* d_output_buffer{};
	//curandState_t* d_random_states{};

	//CUDA_ASSERT(cudaMalloc(&d_output_buffer, sizeof(glm::vec4) * render_width * render_height));
	//CUDA_ASSERT(cudaMalloc(&d_random_states, sizeof(curandState_t) * render_width * render_height));

	darray<glm::vec4> d_output_buffer(render_width * render_height);
	//darray<curandState_t> d_random_states(render_width * render_height);
	darray<cuRandom> rngs(render_width * render_height);

	dobj<Material> default_mat = dobj<MetalAbstract>::Make(glm::vec3(1.0f), 0.1f);

	dim3 threads(8, 8, 1);
	dim3 blocks(ceilDiv(render_width, threads.x), ceilDiv(render_height, threads.y), 1);
	init_random_states<<<blocks, threads>>>(render_width, render_height, 1984, rngs.getPtr());
	CUDA_ASSERT(cudaDeviceSynchronize());


	return Renderer(M{
		render_width,
		render_height,
		samples_per_pixel,
		max_depth,
		cam,
		d_world_ptr,
		std::move(d_output_buffer),
		std::move(rngs),
		std::move(default_mat)
	});
}
Renderer::Renderer(Renderer&& other) : m(std::move(other.m)) {
	//other.m.d_output_buffer = nullptr;
	//other.m.d_random_states = nullptr;
}
Renderer& Renderer::operator=(Renderer&& other) {
	_delete();

	m = std::move(other.m);

	//other.m.d_output_buffer = nullptr;
	//other.m.d_random_states = nullptr;

	return *this;
}
Renderer::~Renderer() {
	_delete();
}

void Renderer::DownloadRenderbuffer(glm::vec4* host_dst) const {
	CUDA_ASSERT(cudaMemcpy(host_dst, m.d_output_buffer.getPtr(), sizeof(glm::vec4) * m.render_width * m.render_height, cudaMemcpyDeviceToHost));
}


struct LaunchParams {
	uint32_t render_width{};
	uint32_t render_height{};
	uint32_t samples_per_pixel{};
	uint32_t max_depth{};
	PinholeCamera cam{};
	HittableList* world{};
	Material* default_mat{};
	glm::vec4* output_buffer{};
	cuRandom* rngs{};
};
__global__ void render_kernel(LaunchParams p);

void Renderer::Render() {
	LaunchParams params{};
	params.render_width = m.render_width;
	params.render_height = m.render_height;
	params.samples_per_pixel = m.samples_per_pixel;
	params.max_depth = m.max_depth;
	params.cam = m.cam;
	params.world = m.d_world_ptr;
	params.default_mat = m.default_mat.getPtr();
	params.output_buffer = m.d_output_buffer.getPtr();
	params.rngs = m.rngs.getPtr();

	// since the program is using virtual functions, the per thread stack size cannot be calculated at compile time,
	// therefore you must manually set a size that should encompass the entire program.
	CUDA_ASSERT(cudaDeviceSetLimit(cudaLimit::cudaLimitStackSize, 2048));

	dim3 threads{ 8, 8, 1 };
	dim3 blocks = dim3(ceilDiv(m.render_width, threads.x), ceilDiv(m.render_height, threads.y), 1);
	render_kernel<<<blocks, threads>>>(params);
	CUDA_ASSERT(cudaDeviceSynchronize());
}

__device__ glm::vec3 sample_world(const Ray& ray, const LaunchParams& p, cuRandom& random_state) {
	Ray cur_ray = ray;
	glm::vec3 accum_attenuation(1.0f);
	//glm::vec3 accum_radiance(0.0f);

	// bounce loop

	for (int i = 0; i < p.max_depth; i++) {
		TraceRecord rec{};

		if (!p.world->ClosestIntersection(cur_ray, rec)) {
			float t = glm::normalize(cur_ray.d).y * 0.5f + 0.5f;
			glm::vec3 sky_radiance = glm::linear_interpolate(glm::vec3(0.1f, 0.2f, 0.4f), glm::vec3(0.9f, 0.9f, 0.99f), t);
			//return accum_attenuation * sky_radiance + accum_radiance;
			return accum_attenuation * sky_radiance;
		}

		// evaluate surface material
		// add emission multiplied by accum_attenuation to accum_radiance

		Ray scattered{};
		glm::vec3 attenuation{};
		if (!rec.mat_ptr->Scatter(cur_ray, rec, random_state, scattered, attenuation)) {
			// ray absorbed
			//return accum_radiance;
			return glm::vec3(0.0f);
		}

		// accumulate material attenuation
		accum_attenuation *= attenuation;
		cur_ray = scattered;

		// offset ray from surface to avoid shadow acne
		cur_ray.o += glm::sign(glm::dot(cur_ray.d, rec.n)) * rec.n * 0.0003f;
	}

	// max bounces exceeded
	//return accum_radiance;
	return glm::vec3(0.0f);
}

__global__ void render_kernel(LaunchParams p) {
	int x = THREAD_GLOBAL_X_ID;
	int y = THREAD_GLOBAL_Y_ID;
	if (x >= p.render_width || y >= p.render_height) return;

	glm::vec2 pixel_size = 1.0f / glm::vec2(p.render_width, p.render_height);

	int gid = y * p.render_width + x;
	cuRandom& random_state = p.rngs[gid];
	glm::vec2 ndc = (glm::vec2(x, y) + glm::vec2(0.5f)) * pixel_size * 2.0f - 1.0f;
	glm::vec3 accumulated_radiance(0.0f);


	glm::vec3 radiance{};

	for (int s = 0; s < p.samples_per_pixel; s++) {
		glm::vec2 rnd_ndc_sample = ndc + glm::cuRandomInUnit<2>(random_state) * pixel_size;

		Ray ray = p.cam.sample_ray(rnd_ndc_sample.x, rnd_ndc_sample.y);

		radiance += sample_world(ray, p, random_state);
	}

	radiance *= 1.0f / p.samples_per_pixel;

	// tone mapping would be ideal but clamping will do for now
	glm::vec3 col = glm::clamp(radiance, 0.0f, 1.0f);
	// gamma correction equivalent to gamma 2.0
	col = glm::sqrt(col);
	glm::vec4 output_color = glm::vec4(col, 1.0f);

	p.output_buffer[gid] = output_color;
}