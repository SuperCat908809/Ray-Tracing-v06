#include "../pch.h"
#include "gl_camera.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/vector_angle.hpp>

using namespace gl_engine;


Camera::Camera(glm::vec3 pos, glm::vec3 ori, glm::vec3 up) {
	position = pos;
	orientation = ori;
	Camera::up = up;
}

glm::mat4 Camera::Matrix(float fov, float aspect, float near, float far) const {

	glm::mat4 view = glm::lookAt(position, position + orientation, up);
	glm::mat4 proj = glm::perspective(glm::radians(fov), aspect, near, far);

	return proj * view;

}