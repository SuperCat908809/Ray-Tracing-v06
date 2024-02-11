#include "Renderer.cuh"
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

	default_mat = std::make_unique<HandledDeviceAbstract<DielectricAbstract>>(glm::vec3(1.0f), 1.5f);

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
	DielectricAbstract* default_mat{};
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
	params.default_mat = default_mat->getPtr();
	params.output_buffer = d_output_buffer;
	params.random_states = d_random_states;

	// since the program is using virtual functions, the per thread stack size cannot be calculated at compile time,
	// therefore you must manually set a size that should encompass the entire program.
	CUDA_ASSERT(cudaDeviceSetLimit(cudaLimit::cudaLimitStackSize, 2048));

	dim3 threads{ 8, 8, 1 };
	dim3 blocks = dim3((render_width + threads.x - 1) / threads.x, (render_height + threads.y - 1) / threads.y, 1);
	kernel<<<blocks, threads>>>(params);
	CUDA_ASSERT(cudaPeekAtLastError());
	CUDA_ASSERT(cudaDeviceSynchronize());
	CUDA_ASSERT(cudaGetLastError());
}

__device__ glm::vec3 ray_color(const Ray& ray, const LaunchParams& p, curandState* local_rand_state) {
	Ray cur_ray = ray;
	glm::vec3 cur_attenuation(1.0f);
	//glm::vec3 cur_radiance(0.0f);

	// bounce loop

	for (int i = 0; i < p.max_depth; i++) {
		TraceRecord rec{};

		if (!p.sphere_list->ClosestIntersection(cur_ray, rec)) {
			float t = glm::normalize(cur_ray.d).y * 0.5f + 0.5f;
			glm::vec3 sky_radiance = glm::linear_interpolate(glm::vec3(0.1f, 0.2f, 0.4f), glm::vec3(0.9f, 0.9f, 0.99f), t);
			return cur_attenuation * sky_radiance;
		}

		// evaluate surface material

		Ray scattered{};
		glm::vec3 attenuation{};
		if (!rec.mat_ptr->Scatter(cur_ray, rec, local_rand_state, scattered, attenuation)) {
			// ray absorbed
			return glm::vec3(0.0f);
		}

		// accumulate material attenuation
		cur_attenuation *= attenuation;
		cur_ray = scattered;

		// offset ray from surface to avoid shadow acne
		if (glm::dot(cur_ray.d, rec.n) > 0) {
			cur_ray.o += rec.n * 0.0003f;
		}
		else {
			cur_ray.o += -rec.n * 0.0003f;
		}
	}

	// max bounces exceeded, terminate sample
	return glm::vec3(0.0f);
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

#if 0
	for (int sample_idx = 0; sample_idx < p.samples_per_pixel; sample_idx++) {
		glm::vec2 rnd_ndc_sample = ndc + glm::cu_random_in_unit_vec<2>(random_state) * pixel_size;
		Ray cur_ray = p.cam.sample_ray(rnd_ndc_sample.x, rnd_ndc_sample.y);
		glm::vec3 cur_attenuation(1.0f);

		for (int depth = 0; depth < p.max_depth; depth++) {
			TraceRecord rec{};

			if (p.sphere_list->ClosestIntersection(cur_ray, rec)) {
				Ray scatter_ray{};
				glm::vec3 attenuation{};

				//Material* mat_ptr = rec.mat_ptr == nullptr ? p.default_mat : rec.mat_ptr;
				Material* mat_ptr = rec.mat_ptr;
				//Material* mat_ptr = p.default_mat;
				//if (RND < 1e-5f) printf("x %4i, y %4i, s %4i : %p\n", x, y, sample_idx, rec.mat_ptr);

				if (mat_ptr->Scatter(cur_ray, rec, random_state, scatter_ray, attenuation)) {
					cur_ray = Ray(scatter_ray.o + scatter_ray.d * 0.001f, scatter_ray.d);
					cur_attenuation *= attenuation;
					continue;
				}
				else {
					// ray absorbed
					break;
				}
			}
			else {
				float t = glm::normalize(cur_ray.d).y * 0.5f + 0.5f;
				glm::vec3 sky_radiance = glm::linear_interpolate(glm::vec3(0.1f, 0.2f, 0.4f), glm::vec3(0.9f, 0.9f, 0.99f), t);
				accumulated_radiance += cur_attenuation * sky_radiance;
				break;
			}
		}
	}
#elif 0
	for (int sample_idx = 0; sample_idx < p.samples_per_pixel; sample_idx++) {
		glm::vec2 rnd_ndc_sample = ndc + glm::cu_random_in_unit_vec<2>(random_state) * pixel_size;
		Ray cur_ray = p.cam.sample_ray(rnd_ndc_sample.x, rnd_ndc_sample.y);
		glm::vec3 cur_attenuation(1.0f);

		int depth = 0;
		for (;depth < p.max_depth; depth++) {
			TraceRecord rec{};

			if (p.sphere_list->ClosestIntersection(cur_ray, rec)) {
				Ray scatter_ray{};
				glm::vec3 attenuation{};

				if (rec.mat_ptr->Scatter(cur_ray, rec, random_state, scatter_ray, attenuation)) {
					// ray scatters from material
					cur_ray = scatter_ray;
					// offset ray from surface to avoid shadow acne
					if (glm::dot(cur_ray.d, rec.n) > 0) {
						cur_ray.o += rec.n * 0.001f;
					}
					else {
						cur_ray.o += -rec.n * 0.001f;
					}
					cur_attenuation *= attenuation;
					continue;
				}
				else {
					// ray does not scatter. possibly absorbed
					//accumulated_radiance += glm::vec3(0.0f); // nothing gets adds
					break;
				}
			}
			else {
				// ray misses all scene geometry
				float t = glm::normalize(cur_ray.d).y * 0.5f + 0.5f;
				glm::vec3 sky_radiance = glm::linear_interpolate(glm::vec3(0.1f, 0.2f, 0.4f), glm::vec3(0.9f, 0.9f, 0.99f), t);
				accumulated_radiance += cur_attenuation * sky_radiance;
				break;
			}
		}

		// If depth == p.max_depth then nothing is added to the radiance accumulation.
		// Identical to ray being absorbed and can no longer scatter
	}
#else
	// sample radiance accumulation
	glm::vec3 col{};

	for (int s = 0; s < p.samples_per_pixel; s++) {
		// uniformly random point in pixel
		glm::vec2 rnd_ndc_sample = ndc + glm::cu_random_in_unit_vec<2>(random_state) * pixel_size;

		// sample ray from camera according to u, v coords
		Ray ray = p.cam.sample_ray(rnd_ndc_sample.x, rnd_ndc_sample.y);

		// gather incoming radiance for ray and add to radiance accumulation
		col += ray_color(ray, p, random_state);
	}

	// average samples by dividing by sample count
	col /= p.samples_per_pixel;
#endif

	//glm::vec3 average_radiance = accumulated_radiance / (float)p.samples_per_pixel;
	glm::vec4 output_color = glm::vec4(col, 1.0f);

	output_color = glm::clamp(output_color, glm::vec4(0), glm::vec4(1));

	output_color[0] = sqrtf(output_color[0]);
	output_color[1] = sqrtf(output_color[1]);
	output_color[2] = sqrtf(output_color[2]);

	p.output_buffer[gid] = output_color;
}