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

#include "cuHostRND.h"

#if 0
struct _SceneDescription {
	std::vector<dobj<Material>> materials;
	std::vector<dobj<Hittable>> spheres;
	darray<Hittable*> sphere_list;
	dobj<HittableList> world_list;
};
#endif

class SceneBook1 {

	aabb world_bounds;
	HittableList* world{ nullptr };
	Hittable** hittable_list{ nullptr };
	std::vector<Hittable*> hittables;
	std::vector<Sphere*> spheres;
	std::vector<MovingSphere*> moving_spheres;
	std::vector<Material*> materials;


	void _delete();

	friend class _Factory;
	SceneBook1() = default;

public:

	class Factory {

		aabb world_bounds;
		HittableList* world;
		Hittable** hittable_list;
		std::vector<Hittable*> hittables;
		std::vector<Sphere*> spheres;
		std::vector<MovingSphere*> moving_spheres;
		std::vector<Material*> materials;

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

class FirstApp {

	struct M {
		uint32_t render_width{}, render_height{};
		MotionBlurCamera cam{};
		glm::vec4* host_output_framebuffer{};
		Renderer renderer;

		SceneBook1 _sceneDesc;
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