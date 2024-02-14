#include "FirstApp.cuh"

#include <stb/stb_image_write.h>

class Scene1Factory {
public:

	class d_MatFactory {
	public:
		__device__ Material* operator()(size_t input) const {
			switch (input) {
			case 0: return new LambertianAbstract(glm::vec3(0.8f, 0.8f, 0.0f));
			case 1: return new LambertianAbstract(glm::vec3(0.1f, 0.2f, 0.5f));
			case 2: return new DielectricAbstract(glm::vec3(1.0f, 1.0f, 1.0f), 1.5f);
			case 3: return new      MetalAbstract(glm::vec3(0.8f, 0.6f, 0.2f), 0.0f);
			default: return nullptr;
			}
		}
	};

	class d_SphereFactory {
		Material** mat_ptrs{};
	public:
		__device__ d_SphereFactory(Material** ptr2) : mat_ptrs(ptr2) {}
		__device__ Hittable* operator()(size_t input) const {
			switch (input) {
			case 0: return new Sphere(glm::vec3( 0.0f, -100.5f, -1.0f), 100.0f, mat_ptrs[0]);
			case 1: return new Sphere(glm::vec3( 0.0f,    0.0f, -1.0f),   0.5f, mat_ptrs[1]);
			case 2: return new Sphere(glm::vec3(-1.0f,    0.0f, -1.0f),   0.5f, mat_ptrs[2]);
			case 3: return new Sphere(glm::vec3(-1.0f,    0.0f, -1.0f),  -0.4f, mat_ptrs[2]);
			case 4: return new Sphere(glm::vec3( 1.0f,    0.0f, -1.0f),   0.5f, mat_ptrs[3]);
			default: return nullptr;
			}
		}
	};

	Scene1Factory(
		HandledDeviceAbstractArray<Material>** materials,
		HandledDeviceAbstractArray<Hittable>** spheres,
		HandledDeviceAbstract<HittableList>** world_list
	) {
		HandledDeviceAbstract<d_MatFactory> matFact{};
		matFact.MakeOnDevice<d_MatFactory>();
		
		(*materials) = new HandledDeviceAbstractArray<Material>(4);
		(*materials)->MakeOnDeviceFactory<d_MatFactory>(4, 0, 0, matFact.getPtr());

		HandledDeviceAbstract<d_SphereFactory> sphereFact((*materials)->getDeviceArrayPtr());
		(*spheres) = new HandledDeviceAbstractArray<Hittable>(5);
		(*spheres)->MakeOnDeviceFactory<d_SphereFactory>(5, 0, 0, sphereFact.getPtr());

		(*world_list) = new HandledDeviceAbstract<HittableList>((*spheres)->getDeviceArrayPtr(), 5);
	}
};

FirstApp::FirstApp() {
	render_width = 1280;
	render_height = 720;

	glm::vec3 lookfrom(-2, 2, 1);
	glm::vec3 lookat(0, 0, -1);
	glm::vec3 up(0, 1, 0);
	float fov = 20.0f;
	float aspect = render_width / (float)render_height;
	cam = PinholeCamera(lookfrom, lookat, up, fov, aspect);

	HandledDeviceAbstractArray<Material>* factory_materials{};
	HandledDeviceAbstractArray<Hittable>* factory_spheres{};
	HandledDeviceAbstract<HittableList>* factory_world{};
	Scene1Factory(&factory_materials, &factory_spheres, &factory_world);

	sphere_materials = std::unique_ptr<HandledDeviceAbstractArray<Material>>(factory_materials);
	world_sphere_list = std::unique_ptr<HandledDeviceAbstractArray<Hittable>>(factory_spheres);
	world_list = std::unique_ptr<HandledDeviceAbstract<HittableList>>(factory_world);

	renderer = std::make_unique<Renderer>(render_width, render_height, 1024, 64, cam, world_list->getPtr());

	CUDA_ASSERT(cudaMallocHost(&host_output_framebuffer, sizeof(glm::vec4) * render_width * render_height));
}
FirstApp::~FirstApp() {
	CUDA_ASSERT(cudaFreeHost(host_output_framebuffer));
}

void write_renderbuffer_png(std::string filepath, uint32_t width, uint32_t height, glm::vec4* data);
void FirstApp::Run() {
	renderer->Render();
	renderer->DownloadRenderbuffer(host_output_framebuffer);
	write_renderbuffer_png("../renders/test_037.png"s, render_width, render_height, host_output_framebuffer);
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