#ifndef SCENE_CLASSES_CUH
#define SCENE_CLASSES_CUH

#include <vector>

#include "../../utilities/cuda_utilities/cuHostRND.h"

#include "aabb.cuh"
#include "hittable.cuh"
#include "HittableList.cuh"
#include "SphereHittable.cuh"
#include "BVH.cuh"


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
	BVH_Handle bvh;

	void _delete();

	SceneBook2BVH(BVH_Handle&& bvh) : bvh(std::move(bvh)) {}

public:

	class Factory {
		std::vector<SphereHandle> sphere_handles;

		cuHostRND host_rnd{ 512,1984 };

		void _populate_world();

	public:

		Factory() = default;

		SceneBook2BVH MakeScene();
	};

	SceneBook2BVH(SceneBook2BVH&& scene);
	SceneBook2BVH& operator=(SceneBook2BVH&& scene);
	~SceneBook2BVH();

	HittableList* getWorldPtr() { return world; }
};


#endif // SCENE_CLASSES_CUH //