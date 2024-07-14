#ifndef OPENGL_APPLICATION_CLASS_H
#define OPENGL_APPLICATION_CLASS_H

#include <inttypes.h>
#include <string>
#include <memory>

#include "gl_engine/gl_shader.h"
#include "gl_engine/gl_texture.h"
#include "gl_engine/gl_mesh.h"
#include "gl_engine/gl_model.h"


class GLFWwindow;
class ImGuiIO;
namespace gl_engine {

class Camera;

class OpenGL_App {
	uint32_t window_width, window_height;
	GLFWwindow* glfw_window;
	ImGuiIO* imgui_io;

	void _delete();

	void _imgui_inputs();
	void _keyboard_inputs();
	void _mouse_inputs();

	OpenGL_App(OpenGL_App&) = delete;
	OpenGL_App& operator=(OpenGL_App&) = delete;


	bool first_ctrl_w = true;
	bool b_widget_open = true;
	bool draw_model = true;
	bool draw_light = true;
	float light_intensity = 3.0f;
	glm::vec4 light_color = glm::vec4(glm::vec3(1.0f), 1.0f);
	glm::vec3 light_pos = glm::vec3(0.0f, 0.8f, 2.0f);
	float light_scale = 0.05f;

	float delta_time = 0.0f;
	float prev_time = 0.0f;

	std::unique_ptr<Model> object_model = nullptr;
	std::unique_ptr<Model> light_model = nullptr;

	std::unique_ptr<Camera> cam;

public:

	OpenGL_App(uint32_t window_width, uint32_t window_height, std::string title);
	~OpenGL_App();
	OpenGL_App(OpenGL_App&&);
	OpenGL_App& operator=(OpenGL_App&&);

	void Run();

}; // OpenGL_App //
} // gl_engine //

#endif OPENGL_APPLICATION_CLASS_H