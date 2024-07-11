#ifndef OPENGL_CAMERA_CLASS_H
#define OPENGL_CAMERA_CLASS_H

#include <glm/glm.hpp>


class GLFWwindow;

namespace gl_engine {
class Camera{
public:
	glm::vec3 position;
	glm::vec3 orientation;
	glm::vec3 up;

	float speed = 0.1f;
	float sensitivity = 100.0f;

	Camera(glm::vec3 pos, glm::vec3 ori, glm::vec3 up);

	glm::mat4 Matrix(float fov, float aspect, float near, float far) const;
	void ControllerInputs(GLFWwindow* window);
};
}

#endif // OPENGL_CAMERA_CLASS_H //