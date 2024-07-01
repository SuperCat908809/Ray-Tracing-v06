#include "pch.h"

#include "rt_engine/ray_data.cuh"
#include "rt_engine/shaders/cu_Cameras.cuh"
#include "rt_engine/geometry/SphereHittable.cuh"


class SphereTest : public testing::Test {
protected:

	std::vector<Sphere> spheres;
	std::vector<int> ground_truth_indices;

	PinholeCamera cam;
	int width = 1280;
	int height = 720;

	void SetUp() override {
		_populate_world();

		glm::vec3 lookfrom(0, 1, -4);
		glm::vec3 lookat(0, 1, 0);
		glm::vec3 up(0, 1, 0);
		float vfov = 90.0f;
		float aspect = width / (float)height;
		cam = PinholeCamera(lookfrom, lookat, up, vfov, aspect);

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int result_index = _pixel_ground_truth(x, y);
				ground_truth_indices.push_back(result_index);
			}
		}
	}

	void _rnd_sphere(cuHostRND& rnd, int a, int b) {
		float choose_mat = rnd.next();
		glm::vec3 center(a + rnd.next(), 0.2f, b + rnd.next());
		Sphere sp = Sphere(center, 0.2f);
		spheres.push_back(sp);

		if (choose_mat < 0.8f) {
			rnd.next();
			rnd.next();
			rnd.next();

			rnd.next();
			rnd.next();
			rnd.next();

			rnd.next();
		}
		else if (choose_mat < 0.95f) {
			rnd.next();
			rnd.next();
			rnd.next();

			rnd.next();
		}
		else {

		}
	}

	void _populate_world() {
		cuHostRND rnd(512, 1984);

		Sphere ground = Sphere(glm::vec3(0, -1000, 0), 1000);
		spheres.push_back(ground);

		for (int a = -11; a < 11; a++) {
			for (int b = -11; b < 11; b++) {
				_rnd_sphere(rnd, a, b);
			}
		}

		Sphere left = Sphere(glm::vec3(-4, 1, 0), 1);
		spheres.push_back(left);

		Sphere center = Sphere(glm::vec3(0, 1, 0), 1);
		spheres.push_back(center);

		Sphere right = Sphere(glm::vec3(4, 1, 0), 1);
		spheres.push_back(right);
	}

	int _pixel_ground_truth(int x, int y) {

		float u = x / (width - 1.0f) * 2 - 1;
		float v = y / (height - 1.0f) * 2 - 1;

		Ray ray = cam.sample_ray(u, v);
		RayPayload rec{};

		int result_index = -1;

		for (int i = 0; i < spheres.size(); i++) {
			float dist = _sphere_closest_intersection(ray, spheres[i].center, spheres[i].radius);
			if (dist < rec.distance) {
				result_index = i;
				rec.distance = dist;
			}
		}

		return result_index;
	}

};



__global__ void _sphere_index_ker(Sphere* spheres, int sphere_count, PinholeCamera cam, int width, int height, int* results) {
	int x_id = blockDim.x * blockIdx.x + threadIdx.x;
	int y_id = blockDim.y * blockIdx.y + threadIdx.y;
	if (x_id >= width || y_id >= height) return;
	int gid = y_id * width + x_id;

	float u = x_id / (width - 1.0f) * 2 - 1;
	float v = y_id / (height - 1.0f) * 2 - 1;

	Ray ray = cam.sample_ray(u, v);
	RayPayload rec{};

	int result_index = -1;

	for (int i = 0; i < sphere_count; i++) {
		float dist = _sphere_closest_intersection(ray, spheres[i].center, spheres[i].radius);
		if (dist < rec.distance) {
			result_index = i;
			rec.distance = dist;
		}
	}

	results[gid] = result_index;
}

TEST_F(SphereTest, DeviceSphereIndexTest) {

	cudaError_t err;

	Sphere* d_spheres;
	err = cudaMalloc((void**)&d_spheres, sizeof(Sphere) * spheres.size());
	ASSERT_EQ(err, cudaSuccess) << "Failed to allocate deivce memory for scene spheres";

	err = cudaMemcpy(d_spheres, spheres.data(), sizeof(Sphere) * spheres.size(), cudaMemcpyHostToDevice);
	ASSERT_EQ(err, cudaSuccess) << "Failed to copy host spheres to device memory";

	int* d_results;
	err = cudaMalloc((void**)&d_results, sizeof(int) * width * height);
	ASSERT_EQ(err, cudaSuccess) << "Failed to allocate results buffer";


	dim3 threads(8, 8, 1);
	dim3 blocks(ceilDiv(width, threads.x), ceilDiv(height, threads.y), 1);
	_sphere_index_ker<<<blocks, threads>>>(d_spheres, spheres.size(), cam, width, height, d_results);
	err = cudaDeviceSynchronize();
	ASSERT_EQ(err, cudaSuccess) << "Kernel failed";


	int* results = new int[width * height];
	err = cudaMemcpy(results, d_results, sizeof(int) * width * height, cudaMemcpyDeviceToHost);
	ASSERT_EQ(err, cudaSuccess) << "Failed to copy results to host";


	for (int y = 0; y < height; y++)
	for (int x = 0; x < width; x++) {
		int gid = y * width + x;
		EXPECT_EQ(results[gid], ground_truth_indices[gid]) << "Pixel " << x << " " << y << " does not match";
	}


	delete[] results;
	cudaFree(d_results);
	cudaFree(d_spheres);

}