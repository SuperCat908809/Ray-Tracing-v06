#ifndef FIRST_APP_CLASS_H
#define FIRST_APP_CLASS_H

#include <inttypes.h>
#include <vector>

#include "dobj.cuh"
#include "darray.cuh"
#include "dAbstracts.cuh"

#include "hittable.cuh"
#include "HittableList.cuh"
#include "material.cuh"
#include "cu_Cameras.cuh"
#include "Renderer.cuh"
#include "SphereHittable.cuh"
#include "BVH.cuh"

#include "cuHostRND.h"
#include "cuda_utils.cuh"


class SceneBook1 {

	aabb world_bounds;
	HittableList* world{ nullptr };
	Hittable** hittable_list{ nullptr };
	std::vector<SphereHandle> sphere_handles;


	void _delete();

	SceneBook1() = default;

public:

	class Factory {

		aabb world_bounds;
		HittableList* world;
		Hittable** hittable_list;
		std::vector<SphereHandle> sphere_handles;

		cuHostRND host_rnd{ 512,1984 };

		void _populate_world();

	public:

		Factory() = default;

		SceneBook1 MakeScene();
	};

	SceneBook1(SceneBook1&& scene);
	SceneBook1& operator=(SceneBook1&& scene);
	~SceneBook1();

	HittableList* getWorldPtr() { return world; }
};


class SceneBook2BVH {

	aabb world_bounds;
	HittableList* world{ nullptr };
	Hittable** hittable_list{ nullptr };
	std::vector<SphereHandle> sphere_handles;
	//std::vector<Hittable*> bvh_nodes;
	BVH_Handle bvh;

	void _delete();

	SceneBook2BVH(BVH_Handle&& bvh) : bvh(std::move(bvh)) {}

public:

	class Factory {

		//aabb world_bounds;
		//HittableList* world;
		//Hittable** hittable_list;
		std::vector<SphereHandle> sphere_handles;
		//std::vector<Hittable*> bvh_nodes;

		cuHostRND host_rnd{ 512,1984 };

		void _populate_world();
		//const Hittable* _build_bvh();
		//const Hittable* _build_bvh_rec(std::vector<std::tuple<aabb, const Hittable*>>& arr, int start, int end);

	public:

		Factory() = default;

		SceneBook2BVH MakeScene();
	};

	SceneBook2BVH(SceneBook2BVH&& scene);
	SceneBook2BVH& operator=(SceneBook2BVH&& scene);
	~SceneBook2BVH();

	HittableList* getWorldPtr() { return world; }
};


class FirstApp {

	struct M {
		uint32_t render_width{}, render_height{};
		MotionBlurCamera cam{};
		glm::vec4* host_output_framebuffer{};
		Renderer renderer;

		SceneBook2BVH _sceneDesc;
	} m;

	FirstApp(M m) : m(std::move(m)) {}

public:

	FirstApp(const FirstApp&) = delete;
	FirstApp& operator=(const FirstApp&) = delete;

	static FirstApp MakeApp();
	FirstApp(FirstApp&& other) : m(std::move(other.m)) {}
	~FirstApp();

	void Run();
};

#endif // FIRST_APP_CLASS_H //