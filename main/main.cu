#include <iostream>
#include "cuError.h"
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