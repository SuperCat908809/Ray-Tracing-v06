#include "pch.h"
#include "openglApp.h"

#include <cuda_runtime.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

#include "utilities/cuda_utilities/cuError.h"


static const char* vert_shader_source = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"uniform float size;\n"
"void main() {\n"
"    gl_Position = vec4(size * aPos, 1.0f);\n"
"}\0";
static const char* frag_shader_source = "#version 330 core\n"
"out vec4 frag_color;\n"
"uniform vec4 color;\n"
"void main() {\n"
"    frag_color = vec4(color);\n"
"}\0";

static float vertices[] = {
	-0.5f, -0.5f * sqrtf(3) / 3, 0.0f,
	 0.5f, -0.5f * sqrtf(3) / 3, 0.0f,
	 0.0f,  0.5f * sqrtf(3) * 2 / 3, 0.0f,
};

void _make_mesh(uint32_t& vao, uint32_t& vbo) {
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)(0));
	glEnableVertexAttribArray(0);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void _make_shader(uint32_t& shader_program) {
	uint32_t vert_shader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vert_shader, 1, &vert_shader_source, nullptr);
	glCompileShader(vert_shader);

	uint32_t frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(frag_shader, 1, &frag_shader_source, nullptr);
	glCompileShader(frag_shader);

	shader_program = glCreateProgram();
	glAttachShader(shader_program, vert_shader);
	glAttachShader(shader_program, frag_shader);
	glLinkProgram(shader_program);

	glDetachShader(shader_program, vert_shader);
	glDetachShader(shader_program, frag_shader);
	glDeleteShader(vert_shader);
	glDeleteShader(frag_shader);
}


bool gui_initialised = false;
bool glad_initialized = false;
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
void _kill_gui() {

	gui_users--;
	if (gui_users > 0) return;

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwTerminate();
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
	
	if (!glad_initialized) {
		gladLoadGL();
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGui::StyleColorsDark();
		ImGui_ImplOpenGL3_Init("#version 330");
		glad_initialized = true;
	}

	imgui_io = &ImGui::GetIO();
	(void)(*imgui_io);
	ImGui_ImplGlfw_InitForOpenGL(glfw_window, true);


	_make_shader(shader_program);
	_make_mesh(vao, vbo);
}
OpenGL_App::~OpenGL_App() {
	_delete();
	_kill_gui();
}

void OpenGL_App::_delete() {
	if (glfw_window != nullptr) {
		glfwDestroyWindow(glfw_window);
	}

	if (shader_program != 0)
		glDeleteProgram(shader_program);
	if (vao != 0)
		glDeleteVertexArrays(1, &vao);
	if (vbo != 0)
		glDeleteBuffers(1, &vbo);

	glfw_window = nullptr;
	shader_program = 0;
	vao = 0;
	vbo = 0;
}
OpenGL_App::OpenGL_App(OpenGL_App&& other) {
	window_width = other.window_width;
	window_height = other.window_height;
	glfw_window = other.glfw_window;
	imgui_io = other.imgui_io;

	shader_program = other.shader_program;
	vao = other.vao;
	vbo = other.vbo;

	other.glfw_window = nullptr;
	other.shader_program = 0;
	other.vao = 0;
	other.vbo = 0;
}

OpenGL_App& OpenGL_App::operator=(OpenGL_App&& other) {
	_delete();

	window_width = other.window_width;
	window_height = other.window_height;
	glfw_window = other.glfw_window;
	imgui_io = other.imgui_io;

	shader_program = other.shader_program;
	vao = other.vao;
	vbo = other.vbo;

	other.glfw_window = nullptr;
	other.shader_program = 0;
	other.vao = 0;
	other.vbo = 0;

	return *this;
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
	ImGui::ColorEdit4("Triangle color", triangle_color, ImGuiColorEditFlags_PickerHueWheel);

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
			glUseProgram(shader_program);
			glUniform1f(glGetUniformLocation(shader_program, "size"), triangle_size);
			glUniform4fv(glGetUniformLocation(shader_program, "color"), 1, triangle_color);

			glBindVertexArray(vao);
			glDrawArrays(GL_TRIANGLES, 0, 3);
		}

		// render ImGUI last so its drawn on top
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());


		// end frame
		glfwSwapBuffers(glfw_window);
		glfwPollEvents();
	}
}