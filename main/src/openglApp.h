#ifndef OPENGL_APPLICATION_CLASS_H
#define OPENGL_APPLICATION_CLASS_H

#include <inttypes.h>
#include <string>
#include <memory>

#include "gl_engine/shader.h"
#include "gl_engine/gl_texture.h"
#include "gl_engine/gl_mesh.h"


class GLFWwindow;
class ImGuiIO;
namespace gl_engine {

class OpenGL_App {
	uint32_t window_width, window_height;
	GLFWwindow* glfw_window;
	ImGuiIO* imgui_io;

	void _delete();

	void _imgui_inputs();
	void _user_inputs();

	OpenGL_App(OpenGL_App&) = delete;
	OpenGL_App& operator=(OpenGL_App&) = delete;


	bool first_ctrl_w = true;
	bool b_widget_open = true;
	bool draw_triangle = true;

	std::unique_ptr<Shader> shader;
	std::unique_ptr<Mesh> model_mesh;
	std::unique_ptr<Texture> model_albedo_texture;

public:

	OpenGL_App(uint32_t window_width, uint32_t window_height, std::string title);
	~OpenGL_App();
	OpenGL_App(OpenGL_App&&);
	OpenGL_App& operator=(OpenGL_App&&);

	void Run();

}; // OpenGL_App //
} // gl_engine //

#endif OPENGL_APPLICATION_CLASS_H