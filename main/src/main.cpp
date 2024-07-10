#include "pch.h"

#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

#include "utilities/cuda_utilities/cuError.h"
#include "FirstApp.h"
#include "openglApp.h"


enum { RT_ENGINE, GL_ENGINE };

int main() {

	switch (GL_ENGINE) {
	case RT_ENGINE: {
		printf("Creating FirstApp object...\n");
		FirstApp app = FirstApp::MakeApp();
		printf("FirstApp object created.\n");

		printf("Running application\n");
		app.Run();
		printf("Application finished\n");

		break;
	}
	case GL_ENGINE: {
		printf("Creating OpenGL_App object...\n");
		OpenGL_App app = OpenGL_App(800, 800, "OpenGL App testing");
		printf("OpenGL_App object created.\n");

		printf("Running application\n");
		app.Run();
		printf("Application finished\n");

		break;
	}
	}

	CUDA_ASSERT(cudaDeviceReset());

	printf("\n\nFinished.\n");

	return 0;
}