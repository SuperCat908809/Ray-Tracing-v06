#include "FirstApp.cuh"

#include <curand.h>

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

class SceneBook1FinaleFactory {

	static std::vector<float> _makeNUniforms(size_t N, size_t seed) {
		curandGenerator_t gen;
		float* d_rnd_uniforms{};
		cudaMalloc(&d_rnd_uniforms, sizeof(float) * N);

		curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);
		curandSetPseudoRandomGeneratorSeed(gen, 1984ull);
		curandGenerateUniform(gen, d_rnd_uniforms, N);

		std::vector<float> rnds{};
		rnds.resize(N);
		cudaMemcpy(rnds.data(), d_rnd_uniforms, sizeof(float) * N, cudaMemcpyDeviceToHost);

		curandDestroyGenerator(gen);
		cudaFree(d_rnd_uniforms);
		return rnds;
	}

	enum MatIdx { lambert, metal, dielec };
	struct LambertParams { glm::vec3 albedo; };
	struct MetalParams { glm::vec3 albedo; float fuzz; };
	struct DielecParams { glm::vec3 albedo; float ior; };
	struct SphereParams { glm::vec3 origin; float radius; MatIdx mat_type; size_t mat_index; };

	class d_LambertFactory {
		LambertParams* p{};
	public:
		__device__ d_LambertFactory(LambertParams* p) : p(p) {}
		__device__ Material* operator()(size_t index) const {
			LambertParams& p2 = p[index];
			return new LambertianAbstract(p2.albedo);
		}
	};
	class d_MetalFactory {
		MetalParams* p{};
	public:
		__device__ d_MetalFactory(MetalParams* p) : p(p) {}
		__device__ Material* operator()(size_t index) const {
			MetalParams& p2 = p[index];
			return new MetalAbstract(p2.albedo, p2.fuzz);
		}
	};
	class d_DielecFactory {
		DielecParams* p{};
	public:
		__device__ d_DielecFactory(DielecParams* p) : p(p) {}
		__device__ Material* operator()(size_t index) const {
			DielecParams& p2 = p[index];
			return new DielectricAbstract(p2.albedo, p2.ior);
		}
	};
	class d_SphereFactory {
		SphereParams* p{};
		Material** mats;
		int lambert_offset{}, metal_offset{}, dielec_offset{};

		int _getMatOffset(MatIdx idx) const {
			switch (idx) {
			case lambert: return lambert_offset;
			case metal: return metal_offset;
			case dielec: return dielec_offset;
			}
		}

	public:
		__device__ d_SphereFactory(SphereParams* p, Material** mats, int lambert_offset, int metal_offset, int dielec_offset)
			: p(p), mats(mats), lambert_offset(lambert_offset), metal_offset(metal_offset), dielec_offset(dielec_offset) {}
		__device__ Hittable* operator()(size_t index) const {
			SphereParams& p2 = p[index];
			int mat_offset = p2.mat_index + _getMatOffset(p2.mat_type);
			return new Sphere(p2.origin, p2.radius, mats[mat_offset]);
		}
	};

	std::vector<LambertParams> lambert_params{};
	std::vector<MetalParams> metal_params{};
	std::vector<DielecParams> dielec_params{};
	std::vector<SphereParams> sphere_params{};

	void _populateWorld() {
		int rnd_offset = 0;
		auto rnds = _makeNUniforms(4096, 1984);

	#define rnd (rnds[rnd_offset++])

		// ground sphere
		lambert_params.push_back({ glm::vec3(0.5f) });
		sphere_params.push_back({ glm::vec3(0,-1000,0), 1000, lambert, 0 });

		for (int a = -11; a < 11; a++) {
			for (int b = -11; b < 11; b++) {
				float choose_mat = rnd;
				glm::vec3 pos(a + 0.9f * rnd, 0.2f, b + 0.9f * rnd);

				if (glm::length(pos - glm::vec3(4, 0.2f, 0)) > 0.9f) {
					if (choose_mat < 0.8f) {
						// diffse
						glm::vec3 albedo = glm::vec3(rnd, rnd, rnd) * glm::vec3(rnd, rnd, rnd);
						lambert_params.push_back({ albedo });
						sphere_params.push_back({ pos, 0.2f, lambert, lambert_params.size() - 1 });
					}
					else if (choose_mat < 0.95f) {
						// metal
						glm::vec3 albedo = glm::vec3(rnd, rnd, rnd) * 0.5f + 0.5f;
						float fuzz = rnd * 0.5f;
						metal_params.push_back({ albedo, fuzz });
						sphere_params.push_back({ pos, 0.2f, metal, metal_params.size() - 1 });
					}
					else {
						// glass
						dielec_params.push_back({ glm::vec3(1.0f), 1.5f });
						sphere_params.push_back({ pos, 0.2f, dielec, dielec_params.size() - 1 });
					}
				}
			}
		}

		dielec_params.push_back({ glm::vec3(1.0f), 1.5f });
		sphere_params.push_back({ glm::vec3(0,1,0), 1.0f, dielec, dielec_params.size() - 1 });

		lambert_params.push_back({ glm::vec3(0.4f, 0.2f, 0.1f) });
		sphere_params.push_back({ glm::vec3(-4, 1, 0), 1.0f, lambert, lambert_params.size() - 1 });

		metal_params.push_back({ glm::vec3(0.7f, 0.6f, 0.5f), 0.0f });
		sphere_params.push_back({ glm::vec3(4,1,0),1.0f, metal, metal_params.size() - 1 });
	}

public:

	SceneBook1FinaleFactory(
		HandledDeviceAbstractArray<Material>** materials,
		HandledDeviceAbstractArray<Hittable>** sphere_list,
		HandledDeviceAbstract<HittableList>** world_list
	) {
		_populateWorld();

		LambertParams* d_lambert_params{};
		cudaMalloc(&d_lambert_params, sizeof(LambertParams) * lambert_params.size());
		cudaMemcpy(d_lambert_params, lambert_params.data(), sizeof(LambertParams) * lambert_params.size(), cudaMemcpyHostToDevice);
		HandledDeviceAbstract<d_LambertFactory>* lambert_factory{};
		lambert_factory->MakeOnDevice<d_LambertFactory>(d_lambert_params);

		MetalParams* d_dielec_params{};
		cudaMalloc(&d_dielec_params, sizeof(MetalParams) * dielec_params.size());
		cudaMemcpy(d_dielec_params, dielec_params.data(), sizeof(MetalParams) * dielec_params.size(), cudaMemcpyHostToDevice);
		HandledDeviceAbstract<d_LambertFactory>* lambert_factory{};
		lambert_factory->MakeOnDevice<d_LambertFactory>(d_dielec_params);

		(*materials) = new HandledDeviceAbstractArray<Material>(sphere_params.size());
		(*materials)->MakeOnDeviceFactory<d_LambertFactory>(dielec_params.size(), lambert_params.size(), 0, lambert_factory->getPtr());


		cudaFree(d_lambert_params);
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

	renderer = std::make_unique<Renderer>(render_width, render_height, 8, 8, cam, world_list->getPtr());

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