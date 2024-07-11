#include "pch.h"
#include "openglApp.h"

#include <cuda_runtime.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

#include "utilities/cuda_utilities/cuError.h"


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

	delete triangle_shader;
	if (triangle_vao != 0)
		glDeleteVertexArrays(1, &triangle_vao);
	if (triangle_ebo != 0)
		glDeleteBuffers(1, &triangle_ebo);
	if (triangle_vbo != 0)
		glDeleteBuffers(1, &triangle_vbo);
	if (popcat_texture != 0)
		glDeleteTextures(1, &popcat_texture);

	glfw_window = nullptr;
	triangle_shader = nullptr;
	triangle_vao = 0;
	triangle_ebo = 0;
	triangle_vbo = 0;
	popcat_texture = 0;
}
OpenGL_App::OpenGL_App(OpenGL_App&& other) {
	window_width = other.window_width;
	window_height = other.window_height;
	glfw_window = other.glfw_window;
	imgui_io = other.imgui_io;

	first_ctrl_w = other.first_ctrl_w;
	b_widget_open = other.b_widget_open;
	draw_triangle = other.draw_triangle;
	triangle_size = other.triangle_size;
	triangle_color = other.triangle_color;

	triangle_shader = other.triangle_shader;
	triangle_vao = other.triangle_vao;
	triangle_ebo = other.triangle_ebo;
	triangle_vbo = other.triangle_vbo;
	popcat_texture = other.popcat_texture;

	other.triangle_shader = nullptr;
	other.glfw_window = nullptr;
	other.triangle_vao = 0;
	other.triangle_ebo = 0;
	other.triangle_vbo = 0;
	other.popcat_texture = 0;
}
OpenGL_App& OpenGL_App::operator=(OpenGL_App&& other) {
	_delete();

	window_width = other.window_width;
	window_height = other.window_height;
	glfw_window = other.glfw_window;
	imgui_io = other.imgui_io;

	first_ctrl_w = other.first_ctrl_w;
	b_widget_open = other.b_widget_open;
	draw_triangle = other.draw_triangle;
	triangle_size = other.triangle_size;
	triangle_color = other.triangle_color;

	triangle_shader = other.triangle_shader;
	triangle_vao = other.triangle_vao;
	triangle_ebo = other.triangle_ebo;
	triangle_vbo = other.triangle_vbo;
	popcat_texture = other.popcat_texture;

	other.triangle_shader = nullptr;
	other.glfw_window = nullptr;
	other.triangle_vao = 0;
	other.triangle_ebo = 0;
	other.triangle_vbo = 0;
	other.popcat_texture = 0;

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
float vertices[] = {
	-0.5f, -0.5f, 0.0f,    0.0f, 0.0f,
	-0.5f,  0.5f, 0.0f,    0.0f, 1.0f,
	 0.5f,  0.5f, 0.0f,    1.0f, 1.0f,
	 0.5f, -0.5f, 0.0f,    1.0f, 0.0f,
};

uint32_t indices[] = {
	0, 1, 2,
	0, 2, 3,
};
}

void _make_mesh(uint32_t& vao, uint32_t& ebo, uint32_t& vbo) {
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(pop_cat::indices), pop_cat::indices, GL_STATIC_DRAW);

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(pop_cat::vertices), pop_cat::vertices, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(0 * sizeof(float)));
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void _load_texture(uint32_t& tex, std::string filename) {
	int width, height, channels;
	stbi_set_flip_vertically_on_load(true);
	uint8_t* image_data = stbi_load(filename.c_str(), &width, &height, &channels, 3);

	glGenTextures(1, &tex);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tex);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data);

	stbi_image_free(image_data);
	glBindTexture(GL_TEXTURE_2D, 0);
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


	triangle_shader = new Shader("resources/shaders/triangle_vert.glsl", "resources/shaders/triangle_frag.glsl");
	_load_texture(popcat_texture, "resources/images/pop_cat.png");
	_make_mesh(triangle_vao, triangle_ebo, triangle_vbo);
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
	ImGui::Checkbox("Draw triangle", &draw_triangle);
	ImGui::SliderFloat("Triangle size", &triangle_size, 0.05f, 2.0f);
	ImGui::ColorEdit4("Triangle color", &triangle_color[0], ImGuiColorEditFlags_PickerHueWheel);

	ImGui::End();
}

void OpenGL_App::_user_inputs() {
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
}

void OpenGL_App::Run() {

	glfwPollEvents();
	glViewport(0, 0, window_width, window_height);

	while (!glfwWindowShouldClose(glfw_window)) {

		glClearColor(0.07f, 0.13f, 0.17f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);


		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		_imgui_inputs();
		if (glfwWindowShouldClose(glfw_window)) break;
		_user_inputs();
		if (glfwWindowShouldClose(glfw_window)) break;


		// render
		if (draw_triangle) {
			triangle_shader->Use();
			glUniform1f(glGetUniformLocation(triangle_shader->id, "size"), triangle_size);
			glUniform4fv(glGetUniformLocation(triangle_shader->id, "color"), 1, &triangle_color[0]);
			glUniform1i(glGetUniformLocation(triangle_shader->id, "tex0"), 0);

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, popcat_texture);

			glBindVertexArray(triangle_vao);
			//glDrawArrays(GL_TRIANGLES, 0, 3);
			glDrawElements(GL_TRIANGLES, sizeof(pop_cat::indices) / sizeof(uint32_t), GL_UNSIGNED_INT, 0);
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