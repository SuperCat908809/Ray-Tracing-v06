#include "Renderer.cuh"

#include <inttypes.h>
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include "utilities/cuda_utilities/cuError.h"
#include "utilities/timers.h"
#include "utilities/glm_utils.h"

//#include "utilities/cuda_utilities/cuda_objects/darray.cuh"
//#include "utilities/cuda_utilities/cuda_objects/dobj.cuh"

#include "rt_engine/ray_data.cuh"

#include "rt_engine/geometry/hittable.cuh"
//#include "rt_engine/geometry/HittableList.cuh"

#include "rt_engine/shaders/cu_Cameras.cuh"
//#include "rt_engine/shaders/cu_Materials.cuh"
#include "rt_engine/shaders/material.cuh"

#include "utilities/cuda_utilities/cuThreadManagement.cuh"


__global__ void init_random_states(uint32_t width, uint32_t height, int seed, cuRandom* rngs) {
	int x = THREAD_GLOBAL_X_ID;
	int y = THREAD_GLOBAL_Y_ID;
	if (x >= width || y >= height) return;
	int gid = y * width + x;
	//curand_init(seed, gid, 0, &random_states[gid]);
	new (rngs + gid) cuRandom(seed + gid, 0, 0);
}

Renderer Renderer::MakeRenderer(
	uint32_t render_width, uint32_t render_height,
	uint32_t samples_per_pixel, uint32_t max_depth,
	const MotionBlurCamera* cam,
	const Hittable* d_world_ptr
) {

	printf("Allocating Renderer framebuffer and random number generators... ");
	//darray<glm::vec4> d_output_buffer(render_width * render_height);
	//darray<cuRandom> rngs(render_width * render_height);

	glm::vec4* d_output_buffer;
	CUDA_ASSERT(cudaMalloc((void**)&d_output_buffer, sizeof(glm::vec4) * render_width * render_height));

	cuRandom* rngs;
	CUDA_ASSERT(cudaMalloc((void**)&rngs, sizeof(cuRandom) * render_width * render_height));

	printf("done.\n");

	//dobj<Material> default_mat = dobj<MetalAbstract>::Make(glm::vec3(1.0f), 0.1f);

	printf("Initialising random states... ");
	dim3 threads(8, 8, 1);
	dim3 blocks(ceilDiv(render_width, threads.x), ceilDiv(render_height, threads.y), 1);
	init_random_states<<<blocks, threads>>>(render_width, render_height, 1984, rngs);
	CUDA_ASSERT(cudaDeviceSynchronize());
	printf("done.\n");


	return Renderer(M{
		render_width,
		render_height,
		samples_per_pixel,
		max_depth,
		cam,
		d_world_ptr,
		d_output_buffer,
		rngs,
		//std::move(default_mat)
	});
}

void Renderer::_delete() {
	CUDA_ASSERT(cudaFree(m.d_output_buffer));
	CUDA_ASSERT(cudaFree(m.rngs));

	m.d_output_buffer = nullptr;
	m.rngs = nullptr;
}
Renderer::Renderer(Renderer&& other) : m(std::move(other.m)) {
	other.m.d_output_buffer = nullptr;
	other.m.rngs = nullptr;
}
Renderer& Renderer::operator=(Renderer&& other) {
	_delete();

	m = std::move(other.m);

	other.m.d_output_buffer = nullptr;
	other.m.rngs = nullptr;

	return *this;
}
Renderer::~Renderer() {
	_delete();
}

void Renderer::DownloadRenderbuffer(glm::vec4* host_dst) const {
	CUDA_ASSERT(cudaMemcpy(host_dst, m.d_output_buffer, sizeof(glm::vec4) * m.render_width * m.render_height, cudaMemcpyDeviceToHost));
}


struct LaunchParams {
	uint32_t render_width{};
	uint32_t render_height{};
	uint32_t samples_per_pixel{};
	uint32_t max_depth{};
	MotionBlurCamera cam{};
	const Hittable* world{};
	//Material* default_mat{};
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
	params.cam = *m.cam;
	params.world = m.d_world_ptr;
	//params.default_mat = m.default_mat.getPtr();
	params.output_buffer = m.d_output_buffer;
	params.rngs = m.rngs;

	// since the program is using virtual functions, the per thread stack size cannot be calculated at compile time,
	// therefore you must manually set a size that should encompass the entire program.
	CUDA_ASSERT(cudaDeviceSetLimit(cudaLimit::cudaLimitStackSize, 2048 * 4));

	printf("Running render kernel...\n");
	cudaTimer render_timer{};
	render_timer.start();

	dim3 threads{ 8, 8, 1 };
	dim3 blocks = dim3(ceilDiv(m.render_width, threads.x), ceilDiv(m.render_height, threads.y), 1);
	render_kernel<<<blocks, threads>>>(params);
	CUDA_ASSERT(cudaDeviceSynchronize());

	render_timer.end();
	printf("Rendering finished in %fms.\n", render_timer.elapsedms());
}

__device__ glm::vec3 sample_world(const Ray& ray, const LaunchParams& p, cuRandom& random_state) {
	Ray cur_ray = ray;
	glm::vec3 accum_attenuation(1.0f);
	//glm::vec3 accum_radiance(0.0f);

	// bounce loop

	for (int i = 0; i < p.max_depth; i++) {
		RayPayload rec{};

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
		if (!rec.material_ptr->Scatter(cur_ray, rec, random_state, scattered, attenuation)) {
			// ray absorbed
			//return accum_radiance;
			return glm::vec3(0.0f);
		}

		// accumulate material attenuation
		accum_attenuation *= attenuation;
		cur_ray = scattered;

		// offset ray from surface to avoid shadow acne
		//cur_ray.o += glm::sign(glm::dot(cur_ray.d, rec.n)) * rec.n * 0.0003f;
		// the material shader should take responsibility for scatter ray offset 
		// temp for now
		cur_ray.o += cur_ray.d * 0.001f;
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

		Ray ray = p.cam.sample_ray(rnd_ndc_sample.x, rnd_ndc_sample.y, random_state);

		radiance += sample_world(ray, p, random_state);
	}

	radiance *= 1.0f / p.samples_per_pixel;

	// tone mapping would be ideal but clamping will do for now
	glm::vec3 col = glm::clamp(radiance, 0.0f, 1.0f);
	// gamma correction equivalent to gamma 2.0
	col = glm::sqrt(col);
	//float exposure = 2.0f;
	//col = 1.0f - glm::exp(-col * exposure);
	glm::vec4 output_color = glm::vec4(col, 1.0f);

	p.output_buffer[gid] = output_color;
}