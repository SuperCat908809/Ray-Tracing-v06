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
		(*materials)->MakeOnDeviceFactoryPtr<d_MatFactory>(4, 0, 0, matFact.getPtr());

		HandledDeviceAbstract<d_SphereFactory> sphereFact((*materials)->getDeviceArrayPtr());
		(*spheres) = new HandledDeviceAbstractArray<Hittable>(5);
		(*spheres)->MakeOnDeviceFactoryPtr<d_SphereFactory>(5, 0, 0, sphereFact.getPtr());

		(*world_list) = new HandledDeviceAbstract<HittableList>((*spheres)->getDeviceArrayPtr(), 5);
	}
};

class SceneBook1FinaleFactory {

	static std::vector<float> _makeNUniforms(size_t N, size_t seed) {
		curandGenerator_t gen;
		float* d_rnd_uniforms{};
		cudaMalloc(&d_rnd_uniforms, sizeof(float) * N);

		curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);
		curandSetPseudoRandomGeneratorSeed(gen, seed);
		curandGenerateUniform(gen, d_rnd_uniforms, N);

		std::vector<float> rnds{};
		rnds.resize(N);
		cudaMemcpy(rnds.data(), d_rnd_uniforms, sizeof(float) * N, cudaMemcpyDeviceToHost);

		curandDestroyGenerator(gen);
		cudaFree(d_rnd_uniforms);
		return rnds;
	}

	template <typename T>
	static T* _copyToDevice(const std::vector<T>& v) {
		T* d_ptr{};
		CUDA_ASSERT(cudaMalloc(&d_ptr, sizeof(T) * v.size()));
		CUDA_ASSERT(cudaMemcpy(d_ptr, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice));
		return d_ptr;
	}

	enum MatIdx { lambert, metal, dielec };
	struct LambertParams { glm::vec3 albedo; };
	struct MetalParams   { glm::vec3 albedo; float   fuzz; };
	struct DielecParams  { glm::vec3 albedo; float    ior; };
	struct SphereParams  { glm::vec3 origin; float radius; MatIdx mat_type; size_t mat_index; };

	void _makeSphere(int a, int b, std::vector<float>& rnds, int& rnd_offset) {
	#define rnd (rnds[rnd_offset++])

		float choose_mat = rnd;
		glm::vec3 pos(a + 0.9f * rnd, 0.2f, b + 0.9f * rnd);

		if (glm::length(pos - glm::vec3(4, 0.2f, 0)) > 0.9f) {
			if (choose_mat < 0.8f) {
				// diffse
				glm::vec3 albedo = glm::vec3(rnd, rnd, rnd) * glm::vec3(rnd, rnd, rnd);
				sphere_params.push_back({ pos, 0.2f, lambert, lambert_params.size() });
				lambert_params.push_back({ albedo });
			}
			else if (choose_mat < 0.95f) {
				// metal
				glm::vec3 albedo = glm::vec3(rnd, rnd, rnd) * 0.5f + 0.5f;
				float fuzz = rnd * 0.5f;
				sphere_params.push_back({ pos, 0.2f, metal, metal_params.size() });
				metal_params.push_back({ albedo, fuzz });
			}
			else {
				// glass
				sphere_params.push_back({ pos, 0.2f, dielec, dielec_params.size() });
				dielec_params.push_back({ glm::vec3(1.0f), 1.5f });
			}
		}
	}

	void _populateWorld() {
		int rnd_offset = 0;
		auto rnds = _makeNUniforms(4096, 1984);

		// ground sphere
		sphere_params.push_back({ glm::vec3(0,-1000,0), 1000, lambert, lambert_params.size()});
		lambert_params.push_back({ glm::vec3(0.5f) });

		for (int a = -11; a < 11; a++) {
			for (int b = -11; b < 11; b++) {
				_makeSphere(a, b, rnds, rnd_offset);
			}
		}

		sphere_params.push_back({ glm::vec3(0, 1, 0), 1.0f, dielec, dielec_params.size() });
		dielec_params.push_back({ glm::vec3(1.0f), 1.5f });

		sphere_params.push_back({ glm::vec3(-4, 1, 0), 1.0f, lambert, lambert_params.size() });
		lambert_params.push_back({ glm::vec3(0.4f, 0.2f, 0.1f) });

		sphere_params.push_back({ glm::vec3(4, 1, 0), 1.0f, metal, metal_params.size() });
		metal_params.push_back({ glm::vec3(0.7f, 0.6f, 0.5f), 0.0f });
	}

	std::vector<LambertParams> lambert_params{};
	std::vector<MetalParams> metal_params{};
	std::vector<DielecParams> dielec_params{};
	std::vector<SphereParams> sphere_params{};

public:

	class d_LambertFactory {
		LambertParams* p{};
	public:
		__host__ __device__ d_LambertFactory(LambertParams* p) : p(p) {}
		__device__ Material* operator()(size_t index) const {
			LambertParams& p2 = p[index];
			return new LambertianAbstract(p2.albedo);
		}
	};
	class d_MetalFactory {
		MetalParams* p{};
	public:
		__host__ __device__ d_MetalFactory(MetalParams* p) : p(p) {}
		__device__ Material* operator()(size_t index) const {
			MetalParams& p2 = p[index];
			return new MetalAbstract(p2.albedo, p2.fuzz);
		}
	};
	class d_DielecFactory {
		DielecParams* p{};
	public:
		__host__ __device__ d_DielecFactory(DielecParams* p) : p(p) {}
		__device__ Material* operator()(size_t index) const {
			DielecParams& p2 = p[index];
			return new DielectricAbstract(p2.albedo, p2.ior);
		}
	};
	class d_SphereFactory {
		SphereParams* p{};
		Material** mats;
		int lambert_offset{}, metal_offset{}, dielec_offset{};

		__device__ int _getMatOffset(MatIdx idx) const {
			switch (idx) {
			case lambert: return lambert_offset;
			case   metal: return   metal_offset;
			case  dielec: return  dielec_offset;
			}
		}

	public:
		__host__ __device__ d_SphereFactory(SphereParams* p, Material** mats, int lambert_offset, int metal_offset, int dielec_offset)
			: p(p), mats(mats), lambert_offset(lambert_offset), metal_offset(metal_offset), dielec_offset(dielec_offset) {}
		__device__ Hittable* operator()(size_t index) const {
			SphereParams& p2 = p[index];
			int mat_offset = p2.mat_index + _getMatOffset(p2.mat_type);
			return new Sphere(p2.origin, p2.radius, mats[mat_offset]);
		}
	};

	SceneBook1FinaleFactory(
		std::unique_ptr<HandledDeviceAbstractArray<Material>>& materials,
		std::unique_ptr<HandledDeviceAbstractArray<Hittable>>& sphere_list,
		std::unique_ptr<HandledDeviceAbstract<HittableList>>& world_list
	) {
		_populateWorld();

		LambertParams* d_lambert_params = _copyToDevice(lambert_params);
		d_LambertFactory lambert_factory(d_lambert_params);

		MetalParams* d_metal_params = _copyToDevice(metal_params);
		d_MetalFactory metal_factory(d_metal_params);

		DielecParams* d_dielec_params = _copyToDevice(dielec_params);
		d_DielecFactory dielec_factory(d_dielec_params);

		size_t lambert_offset = 0;
		size_t   metal_offset = lambert_offset + lambert_params.size();
		size_t  dielec_offset =   metal_offset +   metal_params.size();

		materials = std::make_unique<HandledDeviceAbstractArray<Material>>(sphere_params.size());
		materials->MakeOnDeviceFactory<d_LambertFactory>(lambert_params.size(), lambert_offset, 0, lambert_factory);
		materials->MakeOnDeviceFactory<d_MetalFactory  >(  metal_params.size(),   metal_offset, 0,   metal_factory);
		materials->MakeOnDeviceFactory<d_DielecFactory >( dielec_params.size(),  dielec_offset, 0,  dielec_factory);

		CUDA_ASSERT(cudaFree(d_lambert_params));
		CUDA_ASSERT(cudaFree(d_metal_params  ));
		CUDA_ASSERT(cudaFree(d_dielec_params ));


		SphereParams* d_sphere_params = _copyToDevice(sphere_params);
		d_SphereFactory sphere_factory(d_sphere_params, materials->getDeviceArrayPtr(), lambert_offset, metal_offset, dielec_offset);

		sphere_list = std::make_unique<HandledDeviceAbstractArray<Hittable>>(sphere_params.size());
		sphere_list->MakeOnDeviceFactory<d_SphereFactory>(sphere_params.size(), 0, 0, sphere_factory);

		CUDA_ASSERT(cudaFree(d_sphere_params));


		world_list = std::make_unique<HandledDeviceAbstract<HittableList>>(sphere_list->getDeviceArrayPtr(), sphere_list->getSize());
	}
};

FirstApp::FirstApp() {
	render_width = 1280;
	render_height = 720;

	glm::vec3 lookfrom(13, 2, 3);
	glm::vec3 lookat(0, 0, 0);
	glm::vec3 up(0, 1, 0);
	float fov = 20.0f;
	float aspect = render_width / (float)render_height;
	cam = PinholeCamera(lookfrom, lookat, up, fov, aspect);

	SceneBook1FinaleFactory(sphere_materials, world_sphere_list, world_list);

	renderer = std::make_unique<Renderer>(render_width, render_height, 1024, 32, cam, world_list->getPtr());

	CUDA_ASSERT(cudaMallocHost(&host_output_framebuffer, sizeof(glm::vec4) * render_width * render_height));
}
FirstApp::~FirstApp() {
	CUDA_ASSERT(cudaFreeHost(host_output_framebuffer));
}

void write_renderbuffer_png(std::string filepath, uint32_t width, uint32_t height, glm::vec4* data);
void FirstApp::Run() {
	renderer->Render();
	renderer->DownloadRenderbuffer(host_output_framebuffer);
	write_renderbuffer_png("../renders/test_039.png"s, render_width, render_height, host_output_framebuffer);
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