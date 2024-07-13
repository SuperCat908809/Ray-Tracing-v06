#include "pch.h"
#include "openglApp.h"

#include <memory>
#include <vector>
#include <cuda_runtime.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

#include "utilities/cuda_utilities/cuError.h"

#include "gl_engine/gl_shader.h"
#include "gl_engine/gl_texture.h"
#include "gl_engine/gl_mesh.h"
#include "gl_engine/gl_camera.h"

using namespace gl_engine;


void _init_gui();
void _init_glad();
void _init_imgui();
void _kill_gui();

OpenGL_App::~OpenGL_App() {
	_delete();
	_kill_gui();
}
void OpenGL_App::_delete() {
	if (glfw_window != nullptr) {
		glfwDestroyWindow(glfw_window);
	}

	glfw_window = nullptr;
}
OpenGL_App::OpenGL_App(OpenGL_App&& other) {
	window_width = other.window_width;
	window_height = other.window_height;
	glfw_window = other.glfw_window;
	imgui_io = other.imgui_io;

	first_ctrl_w = other.first_ctrl_w;
	b_widget_open = other.b_widget_open;
	draw_model = other.draw_model;
	draw_light = other.draw_light;
	light_color = other.light_color;
	light_pos = other.light_pos;
	light_scale = other.light_scale;

	model_shader = std::move(other.model_shader);
	model_mesh = std::move(other.model_mesh);
	model_albedo_texture = std::move(other.model_albedo_texture);

	light_shader = std::move(other.light_shader);
	light_mesh = std::move(other.light_mesh);

	other.glfw_window = nullptr;
}
OpenGL_App& OpenGL_App::operator=(OpenGL_App&& other) {
	_delete();

	window_width = other.window_width;
	window_height = other.window_height;
	glfw_window = other.glfw_window;
	imgui_io = other.imgui_io;

	first_ctrl_w = other.first_ctrl_w;
	b_widget_open = other.b_widget_open;
	draw_model = other.draw_model;
	draw_light = other.draw_light;
	light_color = other.light_color;
	light_pos = other.light_pos;
	light_scale = other.light_scale;

	model_shader = std::move(other.model_shader);
	model_mesh = std::move(other.model_mesh);
	model_albedo_texture = std::move(other.model_albedo_texture);

	light_shader = std::move(other.light_shader);
	light_mesh = std::move(other.light_mesh);

	other.glfw_window = nullptr;

	return *this;
}

namespace triangle {
float vertices[] = {
	-0.5f    , -0.5f * sqrtf(3)     / 3, 0.0f,    0.80f, 0.30f, 0.02f,
	 0.5f    , -0.5f * sqrtf(3)     / 3, 0.0f,    0.80f, 0.30f, 0.02f,
	 0.0f    ,  0.5f * sqrtf(3) * 2 / 3, 0.0f,    1.00f, 0.60f, 0.32f,
	-0.5f / 2,  0.5f * sqrtf(3)     / 6, 0.0f,    0.90f, 0.45f, 0.17f,
	 0.5f / 2,  0.5f * sqrtf(3)     / 6, 0.0f,    0.90f, 0.45f, 0.17f,
	 0.0f    , -0.5f * sqrtf(3)     / 3, 0.0f,    0.80f, 0.30f, 0.02f,
};

uint32_t indices[] = {
	0, 3, 5,
	3, 2, 4,
	5, 4, 1
};
}
namespace pop_cat {
std::vector<Vertex> vertices = {
	Vertex{ glm::vec3{-0.5f, -0.5f, 0.0f }, glm::vec3{0,0,0}, glm::vec2{0.0f, 0.0f} },
	Vertex{ glm::vec3{-0.5f,  0.5f, 0.0f }, glm::vec3{0,0,0}, glm::vec2{0.0f, 1.0f} },
	Vertex{ glm::vec3{ 0.5f,  0.5f, 0.0f }, glm::vec3{0,0,0}, glm::vec2{1.0f, 1.0f} },
	Vertex{ glm::vec3{ 0.5f, -0.5f, 0.0f }, glm::vec3{0,0,0}, glm::vec2{1.0f, 0.0f} },
};

std::vector<uint32_t> indices = {
	0, 1, 2,
	0, 2, 3,
};
}
namespace pyramid {
std::vector<Vertex> vertices = {
	Vertex { glm::vec3(-0.5f, 0.0f,  0.5f),    glm::vec3( 0.0f, -1.0f, 0.0f),    glm::vec2(0.0f, 0.0f) },
	Vertex { glm::vec3(-0.5f, 0.0f, -0.5f),    glm::vec3( 0.0f, -1.0f, 0.0f),    glm::vec2(0.0f, 5.0f) },
	Vertex { glm::vec3( 0.5f, 0.0f, -0.5f),    glm::vec3( 0.0f, -1.0f, 0.0f),    glm::vec2(5.0f, 5.0f) },
	Vertex { glm::vec3( 0.5f, 0.0f,  0.5f),    glm::vec3( 0.0f, -1.0f, 0.0f),    glm::vec2(5.0f, 0.0f) },
					       					      
	Vertex { glm::vec3(-0.5f, 0.0f,  0.5f),    glm::vec3(-0.8f, 0.5f,  0.0f),    glm::vec2(0.0f, 0.0f) },
	Vertex { glm::vec3(-0.5f, 0.0f, -0.5f),    glm::vec3(-0.8f, 0.5f,  0.0f),    glm::vec2(5.0f, 0.0f) },
	Vertex { glm::vec3( 0.0f, 0.8f,  0.0f),    glm::vec3(-0.8f, 0.5f,  0.0f),    glm::vec2(2.5f, 5.0f) },
					       					      
	Vertex { glm::vec3(-0.5f, 0.0f, -0.5f),    glm::vec3( 0.0f, 0.5f, -0.8f),    glm::vec2(5.0f, 0.0f) },
	Vertex { glm::vec3( 0.5f, 0.0f, -0.5f),    glm::vec3( 0.0f, 0.5f, -0.8f),    glm::vec2(0.0f, 0.0f) },
	Vertex { glm::vec3( 0.0f, 0.8f,  0.0f),    glm::vec3( 0.0f, 0.5f, -0.8f),    glm::vec2(2.5f, 5.0f) },
					       					      
	Vertex { glm::vec3( 0.5f, 0.0f, -0.5f),    glm::vec3( 0.8f, 0.5f,  0.0f),    glm::vec2(0.0f, 0.0f) },
	Vertex { glm::vec3( 0.5f, 0.0f,  0.5f),    glm::vec3( 0.8f, 0.5f,  0.0f),    glm::vec2(5.0f, 0.0f) },
	Vertex { glm::vec3( 0.0f, 0.8f,  0.0f),    glm::vec3( 0.8f, 0.5f,  0.0f),    glm::vec2(2.5f, 5.0f) },
					       					      
	Vertex { glm::vec3( 0.5f, 0.0f,  0.5f),    glm::vec3( 0.0f, 0.5f,  0.8f),    glm::vec2(5.0f, 0.0f) },
	Vertex { glm::vec3(-0.5f, 0.0f,  0.5f),    glm::vec3( 0.0f, 0.5f,  0.8f),    glm::vec2(0.0f, 0.0f) },
	Vertex { glm::vec3( 0.0f, 0.8f,  0.0f),    glm::vec3( 0.0f, 0.5f,  0.8f),    glm::vec2(2.5f, 5.0f) },
};
std::vector<uint32_t> indices = {
	0, 1, 2,
	0, 2, 3,
	4, 6, 5,
	7, 9, 8,
	10, 12, 11,
	13, 15, 14,
};
}
namespace light_cube {
std::vector<Vertex> vertices = {
	Vertex { glm::vec3(-1, -1,  1), glm::vec3(0, 0, 0), glm::vec2(0, 0) },
	Vertex { glm::vec3(-1,  1,  1), glm::vec3(0, 0, 0), glm::vec2(0, 0) },
	Vertex { glm::vec3( 1,  1,  1), glm::vec3(0, 0, 0), glm::vec2(0, 0) },
	Vertex { glm::vec3( 1, -1,  1), glm::vec3(0, 0, 0), glm::vec2(0, 0) },

	Vertex { glm::vec3(-1, -1, -1), glm::vec3(0, 0, 0), glm::vec2(0, 0) },
	Vertex { glm::vec3(-1,  1, -1), glm::vec3(0, 0, 0), glm::vec2(0, 0) },
	Vertex { glm::vec3( 1,  1, -1), glm::vec3(0, 0, 0), glm::vec2(0, 0) },
	Vertex { glm::vec3( 1, -1, -1), glm::vec3(0, 0, 0), glm::vec2(0, 0) },
};
std::vector<uint32_t> indices = {
	0, 1, 2,
	0, 2, 3,
	7, 6, 5,
	7, 5, 4,

	1, 5, 6,
	1, 6, 2,
	3, 7, 4,
	3, 4, 0,

	0, 4, 5,
	0, 5, 1,
	3, 2, 6,
	3, 6, 7,
};
}

OpenGL_App::OpenGL_App(uint32_t window_width, uint32_t window_height, std::string title) 
	: window_width(window_width), window_height(window_height) {
	
	_init_gui();

	glfw_window = glfwCreateWindow(window_width, window_height, title.c_str(), nullptr, nullptr);
	if (glfw_window == nullptr) {
		// throw error
		printf("Failed to create window.\n");
		glfwTerminate();
		CUDA_ASSERT(cudaDeviceReset());
		exit(-1);
	}
	glfwMakeContextCurrent(glfw_window);
	
	_init_glad();
	_init_imgui();

	imgui_io = &ImGui::GetIO();
	(void)(*imgui_io);
	ImGui_ImplGlfw_InitForOpenGL(glfw_window, true);


	model_shader = std::unique_ptr<Shader>(Shader::LoadFromFiles("resources/shaders/default_v.glsl", "resources/shaders/default_f.glsl"));
	model_albedo_texture = std::unique_ptr<Texture>(Texture::LoadFromImageFile("resources/images/brick.png"));
	model_mesh = std::make_unique<Mesh>(pyramid::vertices, pyramid::indices);

	light_shader = std::unique_ptr<Shader>(Shader::LoadFromFiles("resources/shaders/light_v.glsl", "resources/shaders/light_f.glsl"));
	light_mesh = std::make_unique<Mesh>(light_cube::vertices, light_cube::indices);

	cam = std::make_unique<Camera>(glm::vec3(0, 0.4f, 2), glm::vec3(0, 0, -1), glm::vec3(0, 1, 0));
}


void OpenGL_App::_imgui_inputs() {
	if (!b_widget_open) return;

	ImGui::Begin("My name is window, ImGUI window", &b_widget_open, ImGuiWindowFlags_MenuBar);
	// menu bar
	if (ImGui::BeginMenuBar()) {
		if (ImGui::BeginMenu("File")) {
			if (ImGui::MenuItem("Toggle widget", "Ctrl+W")) { b_widget_open = !b_widget_open; }
			if (ImGui::MenuItem("Close application", "Ctrl+Q")) {
				glfwSetWindowShouldClose(glfw_window, true);
				return;
			}
			ImGui::EndMenu();
		}
		ImGui::EndMenuBar();
	}

	ImGui::Text("Hello there adventurer!");
	ImGui::Checkbox("Draw model", &draw_model);
	ImGui::Checkbox("Draw light", &draw_light);
	ImGui::ColorEdit4("Light color", glm::value_ptr(light_color), ImGuiColorEditFlags_PickerHueWheel);
	ImGui::DragFloat3("Light pos", glm::value_ptr(light_pos), 0.025f, -1.0f, 1.0f);
	ImGui::DragFloat("Light scale", &light_scale, 0.001f, 0.005f, 0.5f);

	ImGui::End();
}

void OpenGL_App::_keyboard_inputs() {
	if (imgui_io->WantCaptureKeyboard) return;

	// close program
	if (glfwGetKey(glfw_window, GLFW_KEY_LEFT_CONTROL) && glfwGetKey(glfw_window, GLFW_KEY_Q)) {
		glfwSetWindowShouldClose(glfw_window, true);
		return;
	}

	// toggle widget
	if (glfwGetKey(glfw_window, GLFW_KEY_LEFT_CONTROL) && glfwGetKey(glfw_window, GLFW_KEY_W)) {
		if (first_ctrl_w) {
			b_widget_open = !b_widget_open;
			first_ctrl_w = false;
		}
	}
	else {
		first_ctrl_w = true;
	}

	cam->KeyboardInputs(glfw_window, delta_time);
}

void OpenGL_App::_mouse_inputs() {
	if (imgui_io->WantCaptureMouse) return;

	cam->MouseInputs(glfw_window, window_width, window_height, delta_time);
}

void OpenGL_App::Run() {

	prev_time = (float)glfwGetTime();

	glfwPollEvents();
	glViewport(0, 0, window_width, window_height);

	glEnable(GL_DEPTH_TEST);

	while (!glfwWindowShouldClose(glfw_window)) {

		glClearColor(0.07f, 0.13f, 0.17f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		float crnt_time = (float)glfwGetTime();
		delta_time = crnt_time - prev_time;
		prev_time = crnt_time;

		_imgui_inputs();
		if (glfwWindowShouldClose(glfw_window)) break;
		_keyboard_inputs();
		if (glfwWindowShouldClose(glfw_window)) break;
		_mouse_inputs();
		if (glfwWindowShouldClose(glfw_window)) break;

		// render
		glm::mat4 cam_matrix = cam->Matrix(45.0f, window_width / (float)window_height, 0.1f, 100.0f);

		if (draw_model) {
			model_shader->Use();
			model_albedo_texture->Bind(0);

			float rotation = (float)glfwGetTime() / 16;
			rotation *= 360;

			glm::mat4 model = glm::mat4(1.0f);
			//model = glm::rotate(model, glm::radians(rotation), glm::vec3(0, 1, 0));


			glUniformMatrix4fv(glGetUniformLocation(model_shader->id, "model_matrix"), 1, GL_FALSE, glm::value_ptr(model));
			glUniformMatrix4fv(glGetUniformLocation(model_shader->id, "camera_matrix"), 1, GL_FALSE, glm::value_ptr(cam_matrix));

			glUniform1i(glGetUniformLocation(model_shader->id, "tex0"), 0);
			glUniform3fv(glGetUniformLocation(model_shader->id, "camera_pos"), 1, glm::value_ptr(cam->position));
			glUniform3fv(glGetUniformLocation(model_shader->id, "light_pos"), 1, glm::value_ptr(light_pos));
			glUniform4fv(glGetUniformLocation(model_shader->id, "light_color"), 1, glm::value_ptr(light_color));

			model_mesh->Draw();
		}

		if (draw_light) {
			light_shader->Use();

			glm::mat4 model = glm::mat4(1.0f);
			model = glm::translate(model, light_pos);
			model = glm::scale(model, glm::vec3(light_scale));

			glUniformMatrix4fv(glGetUniformLocation(light_shader->id, "model_matrix"), 1, GL_FALSE, glm::value_ptr(model));
			glUniformMatrix4fv(glGetUniformLocation(light_shader->id, "camera_matrix"), 1, GL_FALSE, glm::value_ptr(cam_matrix));
			glUniform4fv(glGetUniformLocation(light_shader->id, "light_color"), 1, glm::value_ptr(light_color));

			light_mesh->Draw();
		}

		// render ImGUI last so its drawn on top
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());


		// end frame
		glfwSwapBuffers(glfw_window);
		glfwPollEvents();
	}
}


bool gui_initialised = false;
bool glad_initialized = false;
bool imgui_initialized = false;
int gui_users = 0;
void _init_gui() {

	gui_users++;
	if (gui_initialised) return;

	glfwInit();

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	gui_initialised = true;
}
void _init_glad() {
	if (glad_initialized) return;

	gladLoadGL();

	glad_initialized = true;
}
void _init_imgui() {
	if (imgui_initialized) return;

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::StyleColorsDark();
	ImGui_ImplOpenGL3_Init("#version 330");

	imgui_initialized = true;
}
void _kill_gui() {

	gui_users--;
	if (gui_users > 0) return;

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwTerminate();
}