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

	float speed = 1.0f;
	float sensitivity = 360.0f * 5;
	bool first_click = true;

	Camera(glm::vec3 pos, glm::vec3 ori, glm::vec3 up);

	glm::mat4 Matrix(float fov, float aspect, float near, float far) const;
	void KeyboardInputs(GLFWwindow* window, float delta_time);
	void MouseInputs(GLFWwindow* window, uint32_t window_width, uint32_t window_height, float dt);
};
}

#endif // OPENGL_CAMERA_CLASS_H //