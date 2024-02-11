#include "FirstApp.h"
#include <string>
#include <stb/stb_image_write.h>
#include <glm/glm.hpp>

FirstApp::FirstApp() {
	render_width = 1280;
	render_height = 720;

	glm::vec3 lookfrom(-2, 2, 1);
	glm::vec3 lookat(0, 0, -1);
	glm::vec3 up(0, 1, 0);
	float fov = 20.0f;
	float aspect = render_width / (float)render_height;
	cam = PinholeCamera(lookfrom, lookat, up, fov, aspect);

	ground_mat = std::make_unique<HandledDeviceAbstract<LambertianAbstract>>(glm::vec3(0.8f, 0.8f, 0.0f));
	center_mat = std::make_unique<HandledDeviceAbstract<LambertianAbstract>>(glm::vec3(0.1f, 0.2f, 0.5f));
	left_mat   = std::make_unique<HandledDeviceAbstract<DielectricAbstract>>(glm::vec3(1.0f, 1.0f, 1.0f), 1.5f);
	right_mat  = std::make_unique<HandledDeviceAbstract<     MetalAbstract>>(glm::vec3(0.8f, 0.6f, 0.2f), 0.0f);





	/*
	
	

		Since cuda does not cross link with other translation units during compilation and linking, device code will be recompiled 
		for every translation unit thus even if you make an object on the device, it cannot be used outside of that translation unit.
		Other translation units may have access to the same function but since they are compiled seperately, they have duplicate 
		instances of it and will not accept and device functions from any other translation unit.

		This is why an instance of the abstract material struct, e.g. LambertianAbstract, if created in device code in Renderer.cu, 
		it will execute properly. However, an identical instance of that struct made in FirstApp.cpp's device code cannot be used by 
		Renderer.cu device code because the virtual function generated in FirstApp.cpp unit points to FirstApp.cpp's instance of the 
		function.

		Despite both units having a copy of the function, they cannot access each others compilations leading to this strange error 
		that appears.

		To remedy this, any virtual functions and structs/classes containing them should only be used by the unit that generated them.
		This could be accomplished by a single source file to act as an 'environment' where all of these special case objects and 
		functions can be made using opaque functions that are only implemented by this environment source file.


	
	*/
	
	
	
	
	
	std::vector<Sphere> sphere_data{};
	//sphere_data.push_back(Sphere(glm::vec3(-2,    1, 0),    1.00f, left_mat->getPtr()));
	//sphere_data.push_back(Sphere(glm::vec3(-2,    1, 0), -  0.80f, left_mat->getPtr()));
	//sphere_data.push_back(Sphere(glm::vec3( 0,    1, 0),    1.00f, center_mat->getPtr()));
	//sphere_data.push_back(Sphere(glm::vec3( 2,    1, 0),    1.00f, right_mat->getPtr()));
	//sphere_data.push_back(Sphere(glm::vec3( 0, -100, 0),  100.00f, ground_mat->getPtr()));
	sphere_data.push_back(Sphere(glm::vec3( 0.0f, -100.5f, -1.0f), 100.0f, ground_mat->getPtr()));
	sphere_data.push_back(Sphere(glm::vec3( 0.0f,    0.0f, -1.0f),   0.5f, center_mat->getPtr()));
	sphere_data.push_back(Sphere(glm::vec3(-1.0f,    0.0f, -1.0f),   0.5f,   left_mat->getPtr()));
	sphere_data.push_back(Sphere(glm::vec3(-1.0f,    0.0f, -1.0f),  -0.45f,   left_mat->getPtr()));
	sphere_data.push_back(Sphere(glm::vec3( 1.0f,    0.0f, -1.0f),   0.5f,  right_mat->getPtr()));
	sphere_list = std::make_unique<HittableList<Sphere>>(sphere_data);

	renderer = std::make_unique<Renderer>(render_width, render_height, 1024, 256, cam, sphere_list.get());

	CUDA_ASSERT(cudaMallocHost(&host_output_framebuffer, sizeof(glm::vec4) * render_width * render_height));
}
FirstApp::~FirstApp() {
	CUDA_ASSERT(cudaFreeHost(host_output_framebuffer));
}

void write_renderbuffer_png(std::string filepath, uint32_t width, uint32_t height, glm::vec4* data);
void FirstApp::Run() {
	renderer->Render();
	renderer->DownloadRenderbuffer(host_output_framebuffer);
	write_renderbuffer_png("../renders/test_028.png"s, render_width, render_height, host_output_framebuffer);
}

void write_renderbuffer_png(std::string filepath, uint32_t width, uint32_t height, glm::vec4* data) {
	uint8_t* output_image_data = new uint8_t[width * height * 4];
	for (int i = 0; i < width * height; i++) {
		output_image_data[i * 4 + 0] = static_cast<uint8_t>(data[i][0] * 255.999f);
		output_image_data[i * 4 + 1] = static_cast<uint8_t>(data[i][1] * 255.999f);
		output_image_data[i * 4 + 2] = static_cast<uint8_t>(data[i][2] * 255.999f);
		output_image_data[i * 4 + 3] = static_cast<uint8_t>(data[i][3] * 255.999f);
	}

	stbi_flip_vertically_on_write(true);
	stbi_write_png(filepath.c_str(), width, height, 4, output_image_data, sizeof(uint8_t) * width * 4);
	delete[] output_image_data;
}