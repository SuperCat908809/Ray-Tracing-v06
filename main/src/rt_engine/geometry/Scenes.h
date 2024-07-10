#ifndef SCENE_CLASSES_CUH
#define SCENE_CLASSES_CUH

#include <vector>

//#include "../../utilities/cuda_utilities/cuHostRND.h"

//#include "aabb.cuh"
//#include "hittable.cuh"
//#include "HittableList.cuh"
//#include "SphereHittable.cuh"
//#include "BVH.cuh"


#if 0
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
#endif


class aabb;
class Hittable;
class cuHostRND;
class BVH_Handle;
class SphereHandle;

class SceneBook2BVH {

	SceneBook2BVH();
	void _delete();

	BVH_Handle* bvh;
	aabb* world_bounds;
	std::vector<SphereHandle> sphere_handles;

public:

	~SceneBook2BVH();
	SceneBook2BVH(SceneBook2BVH&& scene);
	SceneBook2BVH& operator=(SceneBook2BVH&& scene);

	class Factory;

	const Hittable* getWorldPtr() const;
};

class SceneBook2BVH::Factory {

	void _delete();
	void _populate_world();

	cuHostRND* host_rnd;
	std::vector<SphereHandle> sphere_handles;

public:

	Factory();
	~Factory();

	Factory(Factory&& factory);
	Factory& operator=(Factory&& factory);

	SceneBook2BVH* MakeScene();
};

#endif // SCENE_CLASSES_CUH //