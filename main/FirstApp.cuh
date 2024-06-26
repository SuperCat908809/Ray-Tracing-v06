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
#include "cuda_utils.cuh"


class SphereHandle {

	aabb bounds;
	Material* material_ptr{};
	Sphere* sphere_ptr{};
	MovingSphere* moving_sphere_ptr{};
	Hittable* hittable_ptr{};
	// only one of the two spheres above will be instantiated

	SphereHandle() = default;

	void _delete() {
		CUDA_ASSERT(cudaFree(material_ptr));
		CUDA_ASSERT(cudaFree(sphere_ptr));
		CUDA_ASSERT(cudaFree(moving_sphere_ptr));
		CUDA_ASSERT(cudaFree(hittable_ptr));

		material_ptr = nullptr;
		sphere_ptr = nullptr;
		moving_sphere_ptr = nullptr;
		hittable_ptr = nullptr;
	}

	SphereHandle(const SphereHandle&) = delete;
	SphereHandle& operator=(const SphereHandle&) = delete;

public:

	SphereHandle(SphereHandle&& sp) {
		bounds = sp.bounds;
		material_ptr = sp.material_ptr;
		sphere_ptr = sp.sphere_ptr;
		moving_sphere_ptr = sp.moving_sphere_ptr;
		hittable_ptr = sp.hittable_ptr;

		sp.material_ptr = nullptr;
		sp.sphere_ptr = nullptr;
		sp.moving_sphere_ptr = nullptr;
		sp.hittable_ptr = nullptr;
	}

	~SphereHandle() {
		_delete();
	}

	template <typename MatType> requires GeoAcceptable<Sphere, MatType>
	static SphereHandle MakeSphere(const Sphere& sphere, MatType* mat_ptr) {
		SphereHandle sp{};
		sp.bounds = getSphereBounds(sphere);
		sp.material_ptr = mat_ptr;
		sp.sphere_ptr = newOnDevice<Sphere>(sphere);
		sp.moving_sphere_ptr = nullptr;
		sp.hittable_ptr = newOnDevice<SphereHittable>(sp.sphere_ptr, mat_ptr);
		return sp;
	}

	template <typename MatType> requires GeoAcceptable<MovingSphere, MatType>
	static SphereHandle MakeMovingSphere(const MovingSphere& sphere, MatType* mat_ptr) {
		SphereHandle sp{};
		sp.bounds = getMovingSphereBounds(sphere);
		sp.material_ptr = mat_ptr;
		sp.sphere_ptr = nullptr;
		sp.moving_sphere_ptr = newOnDevice<MovingSphere>(sphere);
		sp.hittable_ptr = newOnDevice<MovingSphereHittable>(sp.moving_sphere_ptr, mat_ptr);
		return sp;
	}

	const Hittable* getHittablePtr() const { return hittable_ptr; }
	aabb getBounds() const { return bounds; }
};

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
	std::vector<Hittable*> bvh_nodes;


	void _delete();

	SceneBook2BVH() = default;

public:

	class Factory {

		aabb world_bounds;
		HittableList* world;
		Hittable** hittable_list;
		std::vector<SphereHandle> sphere_handles;
		std::vector<Hittable*> bvh_nodes;

		cuHostRND host_rnd{ 512,1984 };

		void _populate_world();
		const Hittable* _build_bvh();
		const Hittable* _build_bvh_rec(std::vector<std::tuple<aabb, const Hittable*>>& arr, int start, int end);

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