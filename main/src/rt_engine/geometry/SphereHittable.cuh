#ifndef CU_GEOMETRY_CLASSES_H
#define CU_GEOMETRY_CLASSES_H

#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "../../utilities/cuda_utilities/cuda_utils.cuh"

#include "../ray_data.cuh"
#include "aabb.cuh"
#include "hittable.cuh"
#include "../shaders/material.cuh"


__device__ inline float _sphere_closest_intersection(const Ray& ray, glm::vec3 center, float radius);

class Sphere : public Geometry {
public:

	struct TraceRecord {
		const Sphere* sphere{};
		glm::vec3 normal{};
	};

	glm::vec3 center;
	float radius;


	__host__ __device__ Sphere() = default;
	__host__ __device__ Sphere(glm::vec3 center, float radius) 
		: center(center), radius(radius) {};

	__host__ __device__ static glm::vec3 getNormal(const Ray& ray, const RayPayload& rec);
};

__host__ __device__ aabb getSphereBounds(const Sphere& sp);

class SphereHittable : public Hittable {
	const Sphere* sphere;
	const Material* mat_ptr;

public:

	template <typename MatType> requires GeoAcceptableMat<Sphere, MatType>
	__device__ SphereHittable(const Sphere* sphere, const MatType* mat_ptr) 
		: sphere(sphere), mat_ptr(mat_ptr) {}

	__device__ virtual bool ClosestIntersection(const Ray& ray, RayPayload& rec) const override;
};


class MovingSphere : public Geometry {
public:

	struct TraceRecord {
		const MovingSphere* moving_sphere{};
		glm::vec3 normal{};
	};

	glm::vec3 center0;
	glm::vec3 center1;
	float radius;

	__host__ __device__ MovingSphere() = default;
	__host__ __device__ MovingSphere(glm::vec3 center0, glm::vec3 center1, float radius) 
		: center0(center0), center1(center1), radius(radius) {}

	__host__ __device__ static glm::vec3 getNormal(const Ray& ray, const RayPayload& rec);
};

__host__ __device__ aabb getMovingSphereBounds(const MovingSphere& sp);

class MovingSphereHittable : public Hittable {
	const MovingSphere* moving_sphere;
	const Material* mat_ptr;

public:

	template <typename MatType> requires GeoAcceptableMat<MovingSphere, MatType>
	__device__ MovingSphereHittable(const MovingSphere* moving_sphere, const MatType* mat_ptr) 
		: moving_sphere(moving_sphere), mat_ptr(mat_ptr) {}

	__device__ virtual bool ClosestIntersection(const Ray& ray, RayPayload& rec) const override;
};


class SphereHandle {

	aabb bounds;
	Material* material_ptr{};
	Sphere* sphere_ptr{};
	MovingSphere* moving_sphere_ptr{};
	Hittable* hittable_ptr{};
	// only one of the two spheres above will be instantiated

	SphereHandle() = default;

	void _delete();

	SphereHandle(const SphereHandle&) = delete;
	SphereHandle& operator=(const SphereHandle&) = delete;


	static SphereHandle _makeSphere(const Sphere& sphere, Material* mat_ptr);
	static SphereHandle _makeMovingSphere(const MovingSphere& sphere, Material* mat_ptr);

public:

	SphereHandle(SphereHandle&& sp);
	SphereHandle& operator=(SphereHandle&& sp);

	~SphereHandle() {
		_delete();
	}

	template <typename MatType> requires GeoAcceptableMat<Sphere, MatType>
	static SphereHandle MakeSphere(const Sphere& sphere, MatType* mat_ptr) {
		SphereHandle sp{};
		sp.bounds = getSphereBounds(sphere);
		sp.material_ptr = mat_ptr;
		sp.sphere_ptr = newOnDevice<Sphere>(sphere);
		sp.moving_sphere_ptr = nullptr;
		sp.hittable_ptr = newOnDevice<SphereHittable>(sp.sphere_ptr, mat_ptr);
		return sp;
	}

	template <typename MatType> requires GeoAcceptableMat<MovingSphere, MatType>
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

#endif // CU_GEOMETRY_CLASSES_H //