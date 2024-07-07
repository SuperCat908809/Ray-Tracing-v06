#include "pch.h"

#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

#include "utilities/cuda_utilities/cuError.h"
#include "FirstApp.cuh"


int nmain() {

	{
		printf("Creating FirstApp object...\n");
		FirstApp app = FirstApp::MakeApp();
		printf("FirstApp object created.\n");

		printf("Running application\n");
		app.Run();
		printf("Application finished\n");
	}

	CUDA_ASSERT(cudaDeviceReset());

	printf("\n\nFinished.\n");

	return 0;
}




const char* vert_shader_source = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"uniform float size;\n"
"void main() {\n"
"    gl_Position = vec4(size * aPos, 1.0f);\n"
"}\0";
const char* frag_shader_source = "#version 330 core\n"
"out vec4 frag_color;\n"
"uniform vec4 color;\n"
"void main() {\n"
"    frag_color = vec4(color);\n"
"}\0";

float vertices[] = {
	-0.5f, -0.5f * sqrtf(3)     / 3, 0.0f,
	 0.5f, -0.5f * sqrtf(3)     / 3, 0.0f,
	 0.0f,  0.5f * sqrtf(3) * 2 / 3, 0.0f,
};

void make_mesh(uint32_t& vao, uint32_t& vbo) {
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

void make_shader(uint32_t& shader_program) {
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

int main() {

	glfwInit();

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(800, 800, "Window test", nullptr, nullptr);
	if (window == nullptr) {
		printf("Failed to create window.\n");
		glfwTerminate();
		CUDA_ASSERT(cudaDeviceReset());
		return -1;
	}
	glfwMakeContextCurrent(window);
	gladLoadGL();


	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330");


	uint32_t shader_program;
	make_shader(shader_program);

	uint32_t vao, vbo;
	make_mesh(vao, vbo);


	glfwPollEvents();
	glViewport(0, 0, 800, 800);

	bool b_widget_open = true;
	bool first_ctrl_w = true;
	bool draw_triangle = true;
	float triangle_size = 1.0f;
	float triangle_color[] = {0.8f, 0.3f, 0.02f, 1.0f};

	while (!glfwWindowShouldClose(window)) {
		// being frame
		// clear screen to default color
		glClearColor(0.07f, 0.13f, 0.17f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);


		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		// process ImGUI first as its the controller

		if (b_widget_open) {
			ImGui::Begin("My name is window, ImGUI window", &b_widget_open, ImGuiWindowFlags_MenuBar);
			// menu bar
			if (ImGui::BeginMenuBar()) {
				if (ImGui::BeginMenu("File")) {
					if (ImGui::MenuItem("Toggle widget", "Ctrl+W")) { b_widget_open = !b_widget_open; }
					if (ImGui::MenuItem("Close application", "Ctrl+Q")) {
						glfwSetWindowShouldClose(window, true);
						continue;
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



		if (!io.WantCaptureKeyboard) {

			// close program
			if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) && glfwGetKey(window, GLFW_KEY_Q)) {
				glfwSetWindowShouldClose(window, true);
				continue;
			}

			// toggle widget
			if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) && glfwGetKey(window, GLFW_KEY_W)) {
				if (first_ctrl_w) {
					b_widget_open = !b_widget_open;
					first_ctrl_w = false;
				}
			}
			else {
				first_ctrl_w = true;
			}

		}


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
		glfwSwapBuffers(window);
		glfwPollEvents();
	}


	glDeleteProgram(shader_program);
	glDeleteVertexArrays(1, &vao);
	glDeleteBuffers(1, &vbo);

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(window);
	glfwTerminate();
	CUDA_ASSERT(cudaDeviceReset());

	printf("\n\nFinished.\n");

	return 0;

}