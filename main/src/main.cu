#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "utilities/cuda_utilities/cuError.h"
#include "FirstApp.cuh"


int main() {

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