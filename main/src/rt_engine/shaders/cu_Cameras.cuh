#ifndef CU_CAMERA_CLASSES_H
#define CU_CAMERA_CLASSES_H

#include <math.h>
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include "cuRandom.cuh"
#include "ray_data.cuh"
#include "glm_utils.h"


struct PinholeCamera {
	glm::vec3 o{}, u{}, v{}, w{};

	__host__ __device__ PinholeCamera() {};
	__host__ __device__ PinholeCamera(glm::vec3 lookfrom, glm::vec3 lookat, glm::vec3 up, float vfov, float aspect_ratio) {
		float theta = glm::radians(vfov);
		float viewport_height = tanf(theta * 0.5f);
		float viewport_width = viewport_height * aspect_ratio;

		o = lookfrom;
		w = glm::normalize(lookat - lookfrom);
		u = glm::normalize(glm::cross(up, w)) * viewport_width;
		v = glm::normalize(glm::cross(w, u)) * viewport_height;
	}

	__host__ __device__ Ray sample_ray(float s, float t) const {
		Ray ray(o, w + u * s + v * t);
		return ray;
	}
};


struct DefocusBlurCamera {
	glm::vec3 o{}, u{}, v{}, w{};
	float viewport_width{}, viewport_height{};
	float lens_radius{}, focus_dist{};

	__host__ __device__ DefocusBlurCamera() {}
	__host__ __device__ DefocusBlurCamera(glm::vec3 lookfrom, glm::vec3 lookat, glm::vec3 up, float vfov, float aspect_ratio, float aperture, float focus_dist) {
		float theta = glm::radians(vfov);
		viewport_height = tanf(theta * 0.5f);
		viewport_width = viewport_height * aspect_ratio;

		o = lookfrom;
		w = glm::normalize(lookat - lookfrom);
		u = glm::normalize(glm::cross(up, w));
		v = glm::normalize(glm::cross(w, u));

		lens_radius = aperture * 0.5f;
		this->focus_dist = focus_dist;
	}

	__device__ Ray sample_ray(float s, float t, cuRandom& rnd) const {
		glm::vec2 disc = glm::cuRandomInUnit<2>(rnd);
		glm::vec3 offset = u * disc.x + v * disc.y;
		offset *= lens_radius;

		glm::vec3 forward = w * focus_dist;
		glm::vec3 hori = u * viewport_width * focus_dist;
		glm::vec3 vert = v * viewport_height * focus_dist;

		return Ray(o + offset, forward + hori * s + vert * t - offset);
	}
};


struct MotionBlurCamera {
	glm::vec3 o, u, v, w;
	float t0, t1;

	__host__ __device__ MotionBlurCamera() : o(), u(), v(), w(), t0(0.0f), t1(1.0f) {}
	__host__ __device__ MotionBlurCamera(glm::vec3 lookfrom, glm::vec3 lookat, glm::vec3 up, float vfov, float aspect_ratio, float time0, float time1) {
		t0 = time0;
		t1 = time1;

		float theta = glm::radians(vfov);
		float viewport_height = tanf(theta * 0.5f);
		float viewport_width = viewport_height * aspect_ratio;

		o = lookfrom;
		w = glm::normalize(lookat - lookfrom);
		u = glm::normalize(glm::cross(up, w)) * viewport_width;
		v = glm::normalize(glm::cross(w, u)) * viewport_height;
	}

	__device__ Ray sample_ray(float s, float t, cuRandom& rnd) const {
		return Ray(o, w + u * s + v * t, glm::mix(t0, t1, rnd.next()));
	}
};

#endif // CU_CAMERA_CLASSES_H //