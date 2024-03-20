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


struct _SceneDescription {
	std::vector<dobj<Material>> materials;
	std::vector<dobj<Hittable>> spheres;
	darray<Hittable*> sphere_list;
	dobj<HittableList> world_list;
};

class FirstApp {

	struct M {
		uint32_t render_width{}, render_height{};
		PinholeCamera cam{};
		glm::vec4* host_output_framebuffer{};
		Renderer renderer;

		_SceneDescription _sceneDesc;
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