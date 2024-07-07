#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <imgui/imgui.h>

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
"void main() {\n"
"    gl_Position = vec4(aPos, 1.0f);\n"
"}\0";
const char* frag_shader_source = "#version 330 core\n"
"out vec4 frag_color;\n"
"void main() {\n"
"    frag_color = vec4(0.8f, 0.3f, 0.02f, 1.0f);\n"
"}\0";

float vertices[] = {
	-0.5f, -0.5f * sqrtf(3)     / 3, 0.0f,
	 0.5f, -0.5f * sqrtf(3)     / 3, 0.0f,
	 0.0f,  0.5f * sqrtf(3) * 2 / 3, 0.0f,
};

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

	glViewport(0, 0, 800, 800);



	uint32_t vert_shader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vert_shader, 1, &vert_shader_source, nullptr);
	glCompileShader(vert_shader);

	uint32_t frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(frag_shader, 1, &frag_shader_source, nullptr);
	glCompileShader(frag_shader);

	uint32_t shader_program = glCreateProgram();
	glAttachShader(shader_program, vert_shader);
	glAttachShader(shader_program, frag_shader);
	glLinkProgram(shader_program);

	glDetachShader(shader_program, vert_shader);
	glDetachShader(shader_program, frag_shader);
	glDeleteShader(vert_shader);
	glDeleteShader(frag_shader);



	uint32_t vao, vbo;

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)(0));
	glEnableVertexAttribArray(0);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);



	glfwPollEvents();

	while (!glfwWindowShouldClose(window)) {
		// being frame
		// clear screen to default color
		glClearColor(0.07f, 0.13f, 0.17f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);


		if (glfwGetKey(window, GLFW_KEY_ESCAPE)) {
			glfwSetWindowShouldClose(window, true);
			continue;
		}


		glUseProgram(shader_program);
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, 3);


		// end frame
		glfwSwapBuffers(window);
		glfwPollEvents();
	}


	glDeleteProgram(shader_program);
	glDeleteVertexArrays(1, &vao);
	glDeleteBuffers(1, &vbo);

	glfwDestroyWindow(window);
	glfwTerminate();
	CUDA_ASSERT(cudaDeviceReset());

	printf("\n\nFinished.\n");

	return 0;

}