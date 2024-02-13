#include "FirstApp.h"

#include <stb/stb_image_write.h>


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

	std::vector<Sphere::ConstructorParams> sphere_data{};
	sphere_data.push_back(Sphere::ConstructorParams(glm::vec3( 0.0f, -100.5f, -1.0f), 100.0f, ground_mat->getPtr()));
	sphere_data.push_back(Sphere::ConstructorParams(glm::vec3( 0.0f,    0.0f, -1.0f),   0.5f, center_mat->getPtr()));
	sphere_data.push_back(Sphere::ConstructorParams(glm::vec3(-1.0f,    0.0f, -1.0f),   0.5f,   left_mat->getPtr()));
	sphere_data.push_back(Sphere::ConstructorParams(glm::vec3(-1.0f,    0.0f, -1.0f),  -0.4f,   left_mat->getPtr()));
	sphere_data.push_back(Sphere::ConstructorParams(glm::vec3( 1.0f,    0.0f, -1.0f),   0.5f,  right_mat->getPtr()));

	world_sphere_list = std::make_unique<HandledDeviceAbstractArray<Hittable>>(sphere_data.size());
	world_sphere_list->MakeOnDeviceVector<Sphere>(sphere_data.size(), 0, sphere_data);

	world_list = std::make_unique<HandledDeviceAbstract<HittableList>>(world_sphere_list->getDeviceArrayPtr(), world_sphere_list->getSize());

	renderer = std::make_unique<Renderer>(render_width, render_height, 16, 8, cam, world_list->getPtr());

	CUDA_ASSERT(cudaMallocHost(&host_output_framebuffer, sizeof(glm::vec4) * render_width * render_height));
}
FirstApp::~FirstApp() {
	CUDA_ASSERT(cudaFreeHost(host_output_framebuffer));
}

void write_renderbuffer_png(std::string filepath, uint32_t width, uint32_t height, glm::vec4* data);
void FirstApp::Run() {
	renderer->Render();
	renderer->DownloadRenderbuffer(host_output_framebuffer);
	write_renderbuffer_png("../renders/test_034.png"s, render_width, render_height, host_output_framebuffer);
}

void write_renderbuffer_png(std::string filepath, uint32_t width, uint32_t height, glm::vec4* data) {
	uint8_t* output_image_data = new uint8_t[width * height * 4];
	for (uint32_t i = 0; i < width * height; i++) {
		output_image_data[i * 4 + 0] = static_cast<uint8_t>(data[i][0] * 255.999f);
		output_image_data[i * 4 + 1] = static_cast<uint8_t>(data[i][1] * 255.999f);
		output_image_data[i * 4 + 2] = static_cast<uint8_t>(data[i][2] * 255.999f);
		output_image_data[i * 4 + 3] = static_cast<uint8_t>(data[i][3] * 255.999f);
	}

	stbi_flip_vertically_on_write(true);
	stbi_write_png(filepath.c_str(), width, height, 4, output_image_data, sizeof(uint8_t) * width * 4);
	delete[] output_image_data;
}