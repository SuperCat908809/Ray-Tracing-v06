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

void Camera::KeyboardInputs(GLFWwindow* window, float dt) {

	float displacement = speed * dt;

	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)) {
		displacement *= 4.0f; 
	}

	if (glfwGetKey(window, GLFW_KEY_W)) {
		position += displacement * orientation;
	}
	if (glfwGetKey(window, GLFW_KEY_S)) {
		position += -displacement * orientation;
	}
	if (glfwGetKey(window, GLFW_KEY_D)) {
		position += displacement * glm::normalize(glm::cross(orientation, up));
	}
	if (glfwGetKey(window, GLFW_KEY_A)) {
		position += -displacement * glm::normalize(glm::cross(orientation, up));
	}
	if (glfwGetKey(window, GLFW_KEY_SPACE)) {
		position += displacement * up;
	}
	if (glfwGetKey(window, GLFW_KEY_C)) {
		position += -displacement * up;
	}
}

void Camera::MouseInputs(GLFWwindow* window, uint32_t window_width, uint32_t window_height, float dt) {
	if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)) {
		if (first_click) {
			glfwSetCursorPos(window, (window_width / 2.0), (window_height / 2.0));
			first_click = false;
		}

		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);

		double mouse_x, mouse_y;
		glfwGetCursorPos(window, &mouse_x, &mouse_y);

		float rot_x = sensitivity * dt * (float)(mouse_y - (window_height / 2)) / (window_height / 2);
		float rot_y = sensitivity * dt * (float)(mouse_x - (window_width / 2)) / (window_width / 2);

		glm::vec3 new_ori = glm::rotate(orientation, glm::radians(-rot_x), glm::normalize(glm::cross(orientation, up)));

		if (glm::angle(new_ori, up) > glm::radians(5.0f) && glm::angle(new_ori, -up) > glm::radians(5.0f)) {
			orientation = new_ori;
		}

		orientation = glm::rotate(orientation, glm::radians(-rot_y), up);

		glfwSetCursorPos(window, (window_width / 2.0), (window_height / 2.0));
	}
	else {
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		first_click = true;
	}
}