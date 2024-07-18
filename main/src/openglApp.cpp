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


OpenGL_App::~OpenGL_App() {
	_delete();
}
void OpenGL_App::_delete() {
	if (glfw_window != nullptr) {
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();

		glfwDestroyWindow(glfw_window);

		glfwTerminate();
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
	light_intensity = other.light_intensity;

	object_model = std::move(other.object_model);
	light_model = std::move(other.light_model);

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
	light_intensity = other.light_intensity;

	object_model = std::move(other.object_model);
	light_model = std::move(other.light_model);

	other.glfw_window = nullptr;

	return *this;
}

OpenGL_App::OpenGL_App(uint32_t window_width, uint32_t window_height, std::string title) 
	: window_width(window_width), window_height(window_height) {
	
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	glfw_window = glfwCreateWindow(window_width, window_height, title.c_str(), nullptr, nullptr);
	if (glfw_window == nullptr) {
		// throw error
		printf("Failed to create window.\n");
		glfwTerminate();
		CUDA_ASSERT(cudaDeviceReset());
		exit(-1);
	}
	glfwMakeContextCurrent(glfw_window);
	
	gladLoadGL();

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::StyleColorsDark();
	ImGui_ImplOpenGL3_Init("#version 330");

	imgui_io = &ImGui::GetIO();
	ImGui_ImplGlfw_InitForOpenGL(glfw_window, true);


	std::shared_ptr<Mesh> object_mesh = std::shared_ptr<Mesh>(Mesh::LoadFromObjFile("resources/models/monkey1.obj"));
	std::shared_ptr<Texture> object_albedo = std::shared_ptr<Texture>(Texture::LoadFromFile("resources/images/planks.png"));
	std::shared_ptr<Texture> object_specular = std::shared_ptr<Texture>(Texture::LoadFromFile("resources/images/planksSpec.png"));

	object_albedo->slot = 0;
	object_specular->slot = 1;

	std::shared_ptr<Shader> object_shader = nullptr;

	try {
		object_shader = std::shared_ptr<Shader>(Shader::LoadFromFiles("resources/shaders/default_v.glsl", "resources/shaders/default_f.glsl"));
	}
	catch (const std::runtime_error& err) {
		printf("Error loading object_shader\n\n%s\n\n", err.what());
	}

	object_model = std::unique_ptr<Model>(Model::MakeFrom(object_mesh, object_shader,
		{ {"tex0", object_albedo}, {"tex1", object_specular} }));


	std::shared_ptr<Mesh> light_mesh = std::shared_ptr<Mesh>(Mesh::LoadFromObjFile("resources/models/cube.obj"));
	std::shared_ptr<Shader> light_shader = nullptr;

	try {
		light_shader = std::shared_ptr<Shader>(Shader::LoadFromFiles("resources/shaders/light_v.glsl", "resources/shaders/light_f.glsl"));
	}
	catch (const std::runtime_error& err) {
		printf("Error loading light_shader\n\n%s\n\n", err.what());
		light_shader = nullptr;
	}

	light_model = std::unique_ptr<Model>(Model::MakeFrom(light_mesh, light_shader, {}));

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

	ImGui::Text("Framerate %.1ffps :: %.3fms", imgui_io->Framerate, imgui_io->DeltaTime * 1000.0f);
	ImGui::Spacing();
	ImGui::Checkbox("Draw model", &draw_model);
	ImGui::Checkbox("Draw light", &draw_light);
	ImGui::Spacing();
	ImGui::SliderFloat("Light intensity", &light_intensity, 0.05f, 10.0f);
	ImGui::ColorEdit4("Light color", glm::value_ptr(light_color), ImGuiColorEditFlags_PickerHueWheel);
	ImGui::DragFloat3("Light pos", glm::value_ptr(light_pos), 0.025f, -2.0f, 2.0f);
	ImGui::DragFloat("Light scale", &light_scale, 0.001f, 0.005f, 0.5f);
	ImGui::Spacing();
	if (ImGui::Button("Reload object shader")) {
		try {
			object_model->shader = std::shared_ptr<Shader>(Shader::LoadFromFiles("resources/shaders/default_v.glsl", "resources/shaders/default_f.glsl"));
		}
		catch (const std::runtime_error& err) {
			printf("Error loading model_shader\n\n%s\n\n", err.what());
			object_model->shader = nullptr;
		}
	}
	if (ImGui::Button("Reload light shader")) {
		try {
			light_model->shader = std::shared_ptr<Shader>(Shader::LoadFromFiles("resources/shaders/light_v.glsl", "resources/shaders/light_f.glsl"));
		}
		catch (const std::runtime_error& err) {
			printf("Error loading model_shader\n\n%s\n\n", err.what());
			light_model->shader = nullptr;
		}
	}

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
	glEnable(GL_CULL_FACE);

	uint32_t fbo;
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

	uint32_t fbo_tex;
	glGenTextures(1, &fbo_tex);
	glBindTexture(GL_TEXTURE_2D, fbo_tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, window_width, window_height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbo_tex, 0);

	uint32_t fbo_rbo;
	glGenRenderbuffers(1, &fbo_rbo);
	glBindRenderbuffer(GL_RENDERBUFFER, fbo_rbo);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, window_width, window_height);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, fbo_rbo);

	auto fbo_status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (fbo_status != GL_FRAMEBUFFER_COMPLETE) {
		std::cout << "Framebuffer error: " << fbo_status << "\n";
		assert(0);
	}

	Mesh* screen_quad = Mesh::LoadFromObjFile("resources/models/screen_quad.obj");
	Shader* post_process_shader = nullptr;
	try {
		post_process_shader = Shader::LoadFromFiles("resources/shaders/post_process_v.glsl", "resources/shaders/post_process_f.glsl");
	}
	catch (...) { post_process_shader = nullptr; }
	float slider = 0.5f;
	bool split_hori = true;
	bool post_process = true;

	while (!glfwWindowShouldClose(glfw_window)) {

		float crnt_time = (float)glfwGetTime();
		delta_time = crnt_time - prev_time;
		prev_time = crnt_time;

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		_imgui_inputs();
		ImGui::Begin("New window");
		ImGui::Checkbox("Use post processsing", &post_process);
		ImGui::Checkbox("Split horizontally", &split_hori);
		ImGui::SliderFloat("Split pos", &slider, 0.0f, 1.0f);
		ImGui::Spacing();
		if (ImGui::Button("Reload post processing shader")) {
			try {
				if (post_process_shader != nullptr) {
					delete post_process_shader;
					post_process_shader = nullptr;
				}

				post_process_shader = Shader::LoadFromFiles("resources/shaders/post_process_v.glsl", "resources/shaders/post_process_f.glsl");
			}
			catch (...) { post_process_shader = nullptr; }
		}
		ImGui::End();
		if (glfwWindowShouldClose(glfw_window)) break;
		_keyboard_inputs();
		if (glfwWindowShouldClose(glfw_window)) break;
		_mouse_inputs();
		if (glfwWindowShouldClose(glfw_window)) break;

		// render
		glBindFramebuffer(GL_FRAMEBUFFER, fbo);

		glClearColor(0.07f, 0.13f, 0.17f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glEnable(GL_DEPTH_TEST);


		glm::mat4 cam_matrix = cam->Matrix(45.0f, window_width / (float)window_height, 0.1f, 100.0f);

		if (draw_model && object_model->shader != nullptr) {
			object_model->shader->Use();

			float rotation = (float)glfwGetTime() / 16;
			rotation *= 360;

			glm::mat4 model = glm::mat4(1.0f);
			//model = glm::rotate(model, glm::radians(rotation), glm::vec3(0, 1, 0));


			glUniformMatrix4fv(glGetUniformLocation(object_model->shader->id, "model_matrix"), 1, GL_FALSE, glm::value_ptr(model));
			glUniformMatrix4fv(glGetUniformLocation(object_model->shader->id, "camera_matrix"), 1, GL_FALSE, glm::value_ptr(cam_matrix));

			glUniform3fv(glGetUniformLocation(object_model->shader->id, "camera_pos"), 1, glm::value_ptr(cam->position));
			glUniform3fv(glGetUniformLocation(object_model->shader->id, "light_pos"), 1, glm::value_ptr(light_pos));
			glUniform4fv(glGetUniformLocation(object_model->shader->id, "light_color"), 1, glm::value_ptr(light_color * light_intensity));

			object_model->Draw();
		}

		if (draw_light && light_model->shader != nullptr) {
			light_model->shader->Use();

			glm::mat4 model = glm::mat4(1.0f);
			model = glm::translate(model, light_pos);
			model = glm::scale(model, glm::vec3(light_scale));

			glUniformMatrix4fv(glGetUniformLocation(light_model->shader->id, "model_matrix"), 1, GL_FALSE, glm::value_ptr(model));
			glUniformMatrix4fv(glGetUniformLocation(light_model->shader->id, "camera_matrix"), 1, GL_FALSE, glm::value_ptr(cam_matrix));
			glUniform4fv(glGetUniformLocation(light_model->shader->id, "light_color"), 1, glm::value_ptr(light_color));

			light_model->Draw();
		}


		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		if (post_process_shader != nullptr && post_process) {
			glDisable(GL_DEPTH_TEST);

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, fbo_tex);

			post_process_shader->Use();
			glUniform1i(glGetUniformLocation(post_process_shader->id, "screen_tex"), 0);
			glUniform1f(glGetUniformLocation(post_process_shader->id, "split"), slider);
			glUniform1i(glGetUniformLocation(post_process_shader->id, "split_hori"), split_hori);
			screen_quad->Draw();

			glEnable(GL_DEPTH_TEST);
		}
		else {
			glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
			glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
			glBlitFramebuffer(0, 0, window_width, window_height, 0, 0, window_width, window_height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
		}


		// render ImGUI last so its drawn on top
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());


		// end frame
		glfwSwapBuffers(glfw_window);
		glfwPollEvents();
	}

	delete post_process_shader;
	delete screen_quad;

	glDeleteFramebuffers(1, &fbo);
	glDeleteTextures(1, &fbo_tex);
	glDeleteRenderbuffers(1, &fbo_rbo);
}