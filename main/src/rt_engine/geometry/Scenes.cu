#include "Scenes.h"

#include <glm/glm.hpp>
#include <vector>

#include "../../utilities/cuda_utilities/cuda_utils.cuh"
#include "../../utilities/cuda_utilities/cuHostRND.h"
#include "../../utilities/timers.h"

#include "aabb.cuh"
#include "hittable.cuh"
#include "SphereHittable.cuh"
#include "BVH.cuh"

#include "../shaders/material.cuh"
#include "../shaders/cu_Materials.cuh"


#if 0
SceneBook1::SceneBook1(SceneBook1&& scene) {
	world_bounds = scene.world_bounds;
	world = scene.world;
	hittable_list = scene.hittable_list;
	sphere_handles = std::move(scene.sphere_handles);

	scene.world = nullptr;
	scene.hittable_list = nullptr;
}

SceneBook1& SceneBook1::operator=(SceneBook1&& scene) {
	_delete();

	world_bounds = scene.world_bounds;
	world = scene.world;
	hittable_list = scene.hittable_list;
	sphere_handles = std::move(scene.sphere_handles);

	scene.world = nullptr;
	scene.hittable_list = nullptr;

	return *this;
}

SceneBook1::~SceneBook1() {
	_delete();
}

void SceneBook1::_delete() {
	CUDA_ASSERT(cudaFree(world));
	CUDA_ASSERT(cudaFree(hittable_list));

	world = nullptr;
	hittable_list = nullptr;
}


void SceneBook1::Factory::_populate_world() {

	Sphere ground_sphere = Sphere(glm::vec3(0, -1000, 0), 1000.0f);
	LambertianAbstract<Sphere>* ground_mat = newOnDevice<LambertianAbstract<Sphere>>(glm::vec3(0.5f));
	SphereHandle ground_handle = SphereHandle::MakeSphere(ground_sphere, ground_mat);
	world_bounds += ground_handle.getBounds();
	sphere_handles.push_back(std::move(ground_handle));

	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {

#define rnd host_rnd.next()

			float choose_mat = rnd;
			glm::vec3 center(a + rnd, 0.2f, b + rnd);

			if (choose_mat < 0.8f) {
				auto material = newOnDevice<LambertianAbstract<MovingSphere>>(glm::vec3(rnd * rnd, rnd * rnd, rnd * rnd));
				glm::vec3 center1 = center + glm::vec3(0, rnd * 0.5f, 0);
				auto moving_sphere = MovingSphere(center, center1, 0.2f);
				auto handle = SphereHandle::MakeMovingSphere(moving_sphere, material);
				world_bounds += handle.getBounds();
				sphere_handles.push_back(std::move(handle));
			}
			else if (choose_mat < 0.95f) {
				auto material = newOnDevice<MetalAbstract<Sphere>>(glm::vec3(0.5f * (1.0f + rnd), 0.5f * (1.0f + rnd), 0.5f * (1.0f + rnd)), 0.5f * rnd);
				auto sphere = Sphere(center, 0.2f);
				auto handle = SphereHandle::MakeSphere(sphere, material);
				world_bounds += handle.getBounds();
				sphere_handles.push_back(std::move(handle));
			}
			else {
				auto material = newOnDevice<DielectricAbstract<Sphere>>(glm::vec3(1.0f), 1.5f);
				auto sphere = Sphere(center, 0.2f);
				auto handle = SphereHandle::MakeSphere(sphere, material);
				world_bounds += handle.getBounds();
				sphere_handles.push_back(std::move(handle));
			}
		}
	}

	Sphere center_sphere = Sphere(glm::vec3(0, 1, 0), 1);
	DielectricAbstract<Sphere>* center_mat = newOnDevice<DielectricAbstract<Sphere>>(glm::vec3(1.0f), 1.5f);
	auto center_handle = SphereHandle::MakeSphere(center_sphere, center_mat);
	world_bounds += center_handle.getBounds();
	sphere_handles.push_back(std::move(center_handle));

	Sphere left_sphere = Sphere(glm::vec3(-4, 1, 0), 1);
	LambertianAbstract<Sphere>* left_mat = newOnDevice<LambertianAbstract<Sphere>>(glm::vec3(0.4f, 0.2f, 0.1f));
	auto left_handle = SphereHandle::MakeSphere(left_sphere, left_mat);
	world_bounds += left_handle.getBounds();
	sphere_handles.push_back(std::move(left_handle));

	Sphere right_sphere = Sphere(glm::vec3(4, 1, 0), 1);
	MetalAbstract<Sphere>* right_mat = newOnDevice<MetalAbstract<Sphere>>(glm::vec3(0.7f, 0.6f, 0.5f), 0);
	auto right_handle = SphereHandle::MakeSphere(right_sphere, right_mat);
	world_bounds += right_handle.getBounds();
	sphere_handles.push_back(std::move(right_handle));
}

SceneBook1 SceneBook1::Factory::MakeScene() {

	printf("Populating world... ");
	_populate_world();
	printf("done.\n");

	printf("Building world's HittableList... ");
	std::vector<const Hittable*> hittable_vec{};
	for (int i = 0; i < sphere_handles.size(); i++) {
		const Hittable* ptr = sphere_handles[i].getHittablePtr();
		hittable_vec.push_back(ptr);
	}

	CUDA_ASSERT(cudaMalloc((void**)&hittable_list, sizeof(Hittable*) * hittable_vec.size()));
	CUDA_ASSERT(cudaMemcpy(hittable_list, hittable_vec.data(), sizeof(Hittable*) * hittable_vec.size(), cudaMemcpyHostToDevice));
	world = newOnDevice<HittableList>(
		const_cast<const Hittable**>(hittable_list),
		(int)hittable_vec.size(),
		world_bounds
	);
	printf("done.\n");

	SceneBook1 scene;

	scene.world_bounds = world_bounds;
	scene.world = world;
	scene.hittable_list = hittable_list;
	scene.sphere_handles = std::move(sphere_handles);

	return scene;
}
#endif

SceneBook2BVH::SceneBook2BVH() = default;
SceneBook2BVH::~SceneBook2BVH() {
	_delete();
}
void SceneBook2BVH::_delete() {
	delete bvh;
	delete world_bounds;
	sphere_handles.clear();

	bvh = nullptr;
	world_bounds = nullptr;
}

SceneBook2BVH::SceneBook2BVH(SceneBook2BVH&& scene) {
	bvh = scene.bvh;
	world_bounds = scene.world_bounds;
	sphere_handles = std::move(scene.sphere_handles);

	scene.bvh = nullptr;
	scene.world_bounds = nullptr;
}
SceneBook2BVH& SceneBook2BVH::operator=(SceneBook2BVH&& scene) {
	_delete();

	bvh = scene.bvh;
	world_bounds = scene.world_bounds;
	sphere_handles = std::move(scene.sphere_handles);

	scene.bvh = nullptr;
	scene.world_bounds = nullptr;

	return *this;
}

const Hittable* SceneBook2BVH::getWorldPtr() const {
	return bvh->getBVHPtr();
}


SceneBook2BVH::Factory::Factory() {
	host_rnd = new cuHostRND(512, 1984);
}
SceneBook2BVH::Factory::~Factory() {
	_delete();
}
void SceneBook2BVH::Factory::_delete() {
	delete host_rnd;
	sphere_handles.clear();

	host_rnd = nullptr;
}

SceneBook2BVH::Factory::Factory(SceneBook2BVH::Factory&& factory) {
	host_rnd = factory.host_rnd;
	sphere_handles = std::move(factory.sphere_handles);

	factory.host_rnd = nullptr;
}
SceneBook2BVH::Factory& SceneBook2BVH::Factory::operator=(SceneBook2BVH::Factory&& factory) {
	_delete();

	host_rnd = factory.host_rnd;
	sphere_handles = std::move(factory.sphere_handles);

	factory.host_rnd = nullptr;

	return *this;
}

void SceneBook2BVH::Factory::_populate_world() {

	Sphere ground_sphere = Sphere(glm::vec3(0, -1000, 0), 1000.0f);
	LambertianAbstract<Sphere>* ground_mat = newOnDevice<LambertianAbstract<Sphere>>(glm::vec3(0.5f));
	SphereHandle ground_handle = SphereHandle::MakeSphere(ground_sphere, ground_mat);
	sphere_handles.push_back(std::move(ground_handle));

	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {

#define rnd host_rnd->next()

			float choose_mat = rnd;
			glm::vec3 center(a + rnd, 0.2f, b + rnd);

			if (choose_mat < 0.8f) {
				auto material = newOnDevice<LambertianAbstract<MovingSphere>>(glm::vec3(rnd * rnd, rnd * rnd, rnd * rnd));
				glm::vec3 center1 = center + glm::vec3(0, rnd * 0.5f, 0);
				auto moving_sphere = MovingSphere(center, center1, 0.2f);
				auto handle = SphereHandle::MakeMovingSphere(moving_sphere, material);
				sphere_handles.push_back(std::move(handle));
			}
			else if (choose_mat < 0.95f) {
				auto material = newOnDevice<MetalAbstract<Sphere>>(glm::vec3(0.5f * (1.0f + rnd), 0.5f * (1.0f + rnd), 0.5f * (1.0f + rnd)), 0.5f * rnd);
				auto sphere = Sphere(center, 0.2f);
				auto handle = SphereHandle::MakeSphere(sphere, material);
				sphere_handles.push_back(std::move(handle));
			}
			else {
				auto material = newOnDevice<DielectricAbstract<Sphere>>(glm::vec3(1.0f), 1.5f);
				auto sphere = Sphere(center, 0.2f);
				auto handle = SphereHandle::MakeSphere(sphere, material);
				sphere_handles.push_back(std::move(handle));
			}
		}
	}

	Sphere center_sphere = Sphere(glm::vec3(0, 1, 0), 1);
	DielectricAbstract<Sphere>* center_mat = newOnDevice<DielectricAbstract<Sphere>>(glm::vec3(1.0f), 1.5f);
	auto center_handle = SphereHandle::MakeSphere(center_sphere, center_mat);
	sphere_handles.push_back(std::move(center_handle));

	Sphere left_sphere = Sphere(glm::vec3(-4, 1, 0), 1);
	LambertianAbstract<Sphere>* left_mat = newOnDevice<LambertianAbstract<Sphere>>(glm::vec3(0.4f, 0.2f, 0.1f));
	auto left_handle = SphereHandle::MakeSphere(left_sphere, left_mat);
	sphere_handles.push_back(std::move(left_handle));

	Sphere right_sphere = Sphere(glm::vec3(4, 1, 0), 1);
	MetalAbstract<Sphere>* right_mat = newOnDevice<MetalAbstract<Sphere>>(glm::vec3(0.7f, 0.6f, 0.5f), 0);
	auto right_handle = SphereHandle::MakeSphere(right_sphere, right_mat);
	sphere_handles.push_back(std::move(right_handle));
}

SceneBook2BVH* SceneBook2BVH::Factory::MakeScene() {

	printf("Populating world... ");
	hostTimer populate_timer{};
	populate_timer.start();

	_populate_world();

	populate_timer.end();
	printf("done in %fms.\n", populate_timer.elapsedms());


	printf("Building BVH... ");
	hostTimer bvh_timer{};
	bvh_timer.start();

	std::vector<std::tuple<aabb, const Hittable*>> objects;
	objects.reserve(sphere_handles.size());
	for (int i = 0; i < sphere_handles.size(); i++) {
		auto bounds = sphere_handles[i].getBounds();
		auto hittable_ptr = sphere_handles[i].getHittablePtr();

		objects.push_back(std::make_tuple(bounds, hittable_ptr));
	}

	BVH_Handle::Factory bvh_factory(objects);
#if 1
	bvh_factory.BuildBVH_TopDown();
#else
	bvh_factory.BuildBVH_BottomUp();
#endif
	BVH_Handle bvh_handle = bvh_factory.MakeHandle();

	bvh_timer.end();
	printf("done in %fms.\n", bvh_timer.elapsedms());

	auto scene = new SceneBook2BVH();

	scene->bvh = new BVH_Handle(std::move(bvh_handle));
	scene->world_bounds = new aabb(scene->bvh->getBounds());
	scene->sphere_handles = std::move(sphere_handles);

	return scene;
}