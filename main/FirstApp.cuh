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
	std::vector<Material*> sphere_materials{};
	std::vector<Hittable*> sphere_hittables{};
	Hittable** sphere_list{};
	HittableList* world_list{};
	aabb bounds{};

	void _delete();

	friend class _Factory;
	SceneBook1() = default;

public:

	class Factory {

		std::vector<Material*> sphere_materials{};
		std::vector<Hittable*> sphere_hittables{};
		aabb bounds{};

		cuHostRND host_rnd{ 512,1984 };

		void _populate_world();

	public:

		Factory() = default;

		SceneBook1 MakeScene();
	};

	SceneBook1(SceneBook1&& scene);
	SceneBook1& operator=(SceneBook1&& scene);
	~SceneBook1();

	HittableList* getWorldPtr() { return world_list; }
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