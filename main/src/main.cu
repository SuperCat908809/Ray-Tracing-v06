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


	while (!glfwWindowShouldClose(window)) {



	}


	glfwDestroyWindow(window);
	glfwTerminate();
	CUDA_ASSERT(cudaDeviceReset());

	printf("\n\nFinished.\n");

	return 0;

}