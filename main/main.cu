#include "cuError.h"
#include "FirstApp.cuh"

int main() {

	{
		FirstApp app = FirstApp::MakeApp();
		app.Run();
	}

	CUDA_ASSERT(cudaDeviceReset());

	return 0;
}