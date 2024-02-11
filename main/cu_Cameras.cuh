#ifndef CU_CAMERA_CLASSES_H
#define CU_CAMERA_CLASSES_H

#include "cu_rtCommon.cuh"

struct PinholeCamera {
	glm::vec3 o{}, u{}, v{}, w{};

	__host__ __device__ PinholeCamera() {};
	__host__ __device__ PinholeCamera(glm::vec3 lookfrom, glm::vec3 lookat, glm::vec3 up, float vfov, float aspect_ratio) {
		float theta = glm::radians(vfov);
		//float viewport_width = tanf(theta * 0.5f);
		//float viewport_height = viewport_width / aspect_ratio;
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

#endif // CU_CAMERA_CLASSES_H //