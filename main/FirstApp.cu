#include "FirstApp.cuh"

#include <inttypes.h>
#include <string>
using namespace std::string_literals;
#include <iostream>
#include <vector>

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <stb/stb_image_write.h>

#include "cuError.h"

#include "dobj.cuh"
#include "darray.cuh"
#include "dAbstracts.cuh"

#include "hittable.cuh"
#include "SphereHittable.cuh"
#include "HittableList.cuh"

#include "material.cuh"
#include "cu_Materials.cuh"

#include "cu_Cameras.cuh"
#include "Renderer.cuh"

#include "cuHostRND.h"


#if 0
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
			case 0: return new SphereHittable(glm::vec3( 0.0f, -100.5f, -1.0f), 100.0f, mat_ptrs[0]);
			case 1: return new SphereHittable(glm::vec3( 0.0f,    0.0f, -1.0f),   0.5f, mat_ptrs[1]);
			case 2: return new SphereHittable(glm::vec3(-1.0f,    0.0f, -1.0f),   0.5f, mat_ptrs[2]);
			case 3: return new SphereHittable(glm::vec3(-1.0f,    0.0f, -1.0f),  -0.4f, mat_ptrs[2]);
			case 4: return new SphereHittable(glm::vec3( 1.0f,    0.0f, -1.0f),   0.5f, mat_ptrs[3]);
			default: return nullptr;
			}
		}
	};

	Scene1Factory(
		dAbstractArray<Material>** materials,
		dAbstractArray<Hittable>** spheres,
		dAbstract<HittableList>** world_list
	) {
		dAbstract<d_MatFactory> matFact{};
		matFact.MakeOnDevice<d_MatFactory>();
		
		(*materials) = new dAbstractArray<Material>(4);
		(*materials)->MakeOnDeviceFactoryPtr<d_MatFactory>(4, 0, 0, matFact.getPtr());

		dAbstract<d_SphereFactory> sphereFact((*materials)->getDeviceArrayPtr());
		(*spheres) = new dAbstractArray<Hittable>(5);
		(*spheres)->MakeOnDeviceFactoryPtr<d_SphereFactory>(5, 0, 0, sphereFact.getPtr());

		(*world_list) = new dAbstract<HittableList>((*spheres)->getDeviceArrayPtr(), 5);
	}
};
#endif

#if 0
class SceneBook1FinaleFactory {

	//static std::vector<float> _makeNUniforms(size_t N, size_t seed) {
	//	curandGenerator_t gen;
	//	float* d_rnd_uniforms{};
	//	cudaMalloc(&d_rnd_uniforms, sizeof(float) * N);

	//	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);
	//	curandSetPseudoRandomGeneratorSeed(gen, seed);
	//	curandGenerateUniform(gen, d_rnd_uniforms, N);

	//	std::vector<float> rnds{};
	//	rnds.resize(N);
	//	cudaMemcpy(rnds.data(), d_rnd_uniforms, sizeof(float) * N, cudaMemcpyDeviceToHost);

	//	curandDestroyGenerator(gen);
	//	cudaFree(d_rnd_uniforms);
	//	return rnds;
	//}

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

	void _makeSphere(int a, int b) {
	#define rnd (host_rnd.next())

		float choose_mat = rnd;
		glm::vec3 pos(a + 0.9f * rnd, 0.2f, b + 0.9f * rnd);

		if (glm::length(pos - glm::vec3(4, 0.2f, 0)) > 0.9f) {
			if (choose_mat < 0.8f) {
				// diffuse
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
		// ground sphere
		sphere_params.push_back({ glm::vec3(0,-1000,0), 1000, lambert, lambert_params.size()});
		lambert_params.push_back({ glm::vec3(0.5f) });

		for (int a = -11; a < 11; a++) {
			for (int b = -11; b < 11; b++) {
				_makeSphere(a, b);
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
	std::vector<  MetalParams>   metal_params{};
	std::vector< DielecParams>  dielec_params{};
	std::vector< SphereParams>  sphere_params{};
	cuHostRND host_rnd{ 512, 1984 };

	SceneBook1FinaleFactory() = default;

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
			return new SphereHittable(p2.origin, p2.radius, mats[mat_offset]);
		}
	};

	static _SceneDescription MakeScene() {
		SceneBook1FinaleFactory factory{};

		factory._populateWorld();

		LambertParams* d_lambert_params = _copyToDevice(factory.lambert_params);
		d_LambertFactory lambert_factory(d_lambert_params);
		d_LambertFactory* d_lambert_factory{ nullptr };
		CUDA_ASSERT(cudaMalloc((void**)&d_lambert_factory, sizeof(d_LambertFactory)));
		CUDA_ASSERT(cudaMemcpy(d_lambert_factory, &lambert_factory, sizeof(d_LambertFactory), cudaMemcpyHostToDevice));

		MetalParams* d_metal_params = _copyToDevice(factory.metal_params);
		d_MetalFactory metal_factory(d_metal_params);
		d_MetalFactory* d_metal_factory{ nullptr };
		CUDA_ASSERT(cudaMalloc((void**)&d_metal_factory, sizeof(d_MetalFactory)));
		CUDA_ASSERT(cudaMemcpy(d_metal_factory, &metal_factory, sizeof(d_MetalFactory), cudaMemcpyHostToDevice));

		DielecParams* d_dielec_params = _copyToDevice(factory.dielec_params);
		d_DielecFactory dielec_factory(d_dielec_params);
		d_DielecFactory* d_dielec_factory{ nullptr };
		CUDA_ASSERT(cudaMalloc((void**)&d_dielec_factory, sizeof(d_DielecFactory)));
		CUDA_ASSERT(cudaMemcpy(d_dielec_factory, &dielec_factory, sizeof(d_DielecFactory), cudaMemcpyHostToDevice));

		size_t lambert_offset = 0;
		size_t   metal_offset = lambert_offset + factory.lambert_params.size();
		size_t  dielec_offset = metal_offset + factory.metal_params.size();

		dAbstractArray<Material> materials(factory.sphere_params.size());
		materials.MakeOnDeviceFactory<d_LambertFactory>(factory.lambert_params.size(), lambert_offset, 0, d_lambert_factory);
		materials.MakeOnDeviceFactory<d_MetalFactory  >(factory.  metal_params.size(),   metal_offset, 0,   d_metal_factory);
		materials.MakeOnDeviceFactory<d_DielecFactory >(factory. dielec_params.size(),  dielec_offset, 0,  d_dielec_factory);

		CUDA_ASSERT(cudaFree(d_lambert_factory));
		CUDA_ASSERT(cudaFree(d_metal_factory  ));
		CUDA_ASSERT(cudaFree(d_dielec_factory ));

		CUDA_ASSERT(cudaFree(d_lambert_params));
		CUDA_ASSERT(cudaFree(d_metal_params  ));
		CUDA_ASSERT(cudaFree(d_dielec_params ));


		SphereParams* d_sphere_params = _copyToDevice(factory.sphere_params);
		d_SphereFactory sphere_factory(d_sphere_params, materials.getDeviceArrayPtr(), lambert_offset, metal_offset, dielec_offset);
		d_SphereFactory* d_sphere_factory{ nullptr };
		CUDA_ASSERT(cudaMalloc((void**)&d_sphere_factory, sizeof(d_SphereFactory)));
		CUDA_ASSERT(cudaMemcpy(d_sphere_factory, &sphere_factory, sizeof(d_SphereFactory), cudaMemcpyHostToDevice));

		dAbstractArray<Hittable> sphere_list(factory.sphere_params.size());
		sphere_list.MakeOnDeviceFactory<d_SphereFactory>(factory.sphere_params.size(), 0, 0, d_sphere_factory);

		CUDA_ASSERT(cudaFree(d_sphere_factory));
		CUDA_ASSERT(cudaFree(d_sphere_params));


		dAbstract<HittableList> world_list = dAbstract<HittableList>::MakeAbstract<HittableList>(sphere_list.getDeviceArrayPtr(), sphere_list.getLength());

		return _SceneDescription{
			std::move(materials),
			std::move(sphere_list),
			std::move(world_list)
		};
	}
};
#else

#if 0
class MaterialCollection {
	dAbstractArray<Material> material_ptrs;
	dmemory material_data;
};

class MaterialCollectionFactory {
public:
	enum MaterialID { LAMBERTIAN, METAL, DIELECTRIC };
	struct MaterialParams {
		union {
			struct { MaterialID matID; } matID;
			struct { MaterialID matID; glm::vec3 albedo;             } lambert;
			struct { MaterialID matID; glm::vec3 albedo; float fuzz; }   metal;
			struct { MaterialID matID; glm::vec3 albedo; float  ior; }  dielec;
		};

		__device__ Material* MakeMaterial(Material* location) const {
			switch (matID.matID) {
			case LAMBERTIAN: return new (location) LambertianAbstract(lambert.albedo             );
			case      METAL: return new (location)      MetalAbstract(  metal.albedo,  metal.fuzz);
			case DIELECTRIC: return new (location) DielectricAbstract( dielec.albedo, dielec.ior );
			}
		}
	};

	class MaterialFactory {
		MaterialParams* p;
		Material* reserved_location_root;
		size_t* reserved_locations;
	public:
		__host__ __device__ MaterialFactory(MaterialParams* p, Material* reserved_location_root, size_t* reserved_locations) :
			p(p),
			reserved_location_root(reserved_location_root),
			reserved_locations(reserved_locations) {}
		__device__ Material* operator()(size_t index) const { return p[index].MakeMaterial(reserved_location_root + reserved_locations[index]); }
	};

private:
	MaterialCollectionFactory(const MaterialCollectionFactory&) = delete;
	MaterialCollectionFactory& operator=(const MaterialCollectionFactory&) = delete;

	std::vector<MaterialParams> queued_materials;
public:


	/*
	==== ==== ==== ==== ==== ==== ==== ==== ==== ====
	== All constructed materials will be placed in the same buffer in device memory. This will be achived using the placement new operator.
	== On the host, when a material is queued, it should be given a pointer offset that indicates a location after all previous materials 
	==   and meets the classes alignment requirements.
	== This offset will be added to the pointer that indicates the start of the buffer which will only be known at runtime.
	==
	== The dAbstractArray class should be told to only call the destructor of the materials, not delete which will corrupt memory
	==== ==== ==== ==== ==== ==== ==== ==== ==== ====
	*/



	MaterialCollectionFactory() = default;

	size_t QueueLambert(glm::vec3 albedo            ) { MaterialParams p { .lambert = { LAMBERTIAN, albedo       } }; queued_materials.push_back(p); return queued_materials.size() - 1; }
	size_t   QueueMetal(glm::vec3 albedo, float fuzz) { MaterialParams p { .metal   = {      METAL, albedo, fuzz } }; queued_materials.push_back(p); return queued_materials.size() - 1; }
	size_t  QueueDielec(glm::vec3 albedo, float  ior) { MaterialParams p { .dielec  = { DIELECTRIC, albedo,  ior } }; queued_materials.push_back(p); return queued_materials.size() - 1; }

	MaterialCollection MakeCollection() {
		darray<MaterialParams> params(queued_materials);
		auto material_factory = dobj<MaterialFactory>::Make(params.getPtr());

		dAbstractArray<Material> materials(queued_materials.size());
		materials.MakeOnDeviceFactory(queued_materials.size(), 0, 0, material_factory.getPtr());
	}
};
#endif

class SceneBook1FinaleFactory {
public:
	//enum MaterialID { LAMBERTIAN, METAL, DIELECTRIC };
	//struct MaterialParams {
	//	MaterialID matID;
	//	union {
	//		struct { glm::vec3 albedo; }				lambert;
	//		struct { glm::vec3 albedo; float fuzz; }	  metal;
	//		struct { glm::vec3 albedo; float  ior; }	 dielec;
	//	};

	//	static MaterialParams MakeLambert(glm::vec3 albedo            ) { MaterialParams p; p.matID = LAMBERTIAN; p.lambert = { albedo       }; return p; }
	//	static MaterialParams   MakeMetal(glm::vec3 albedo, float fuzz) { MaterialParams p; p.matID =      METAL; p.metal   = { albedo, fuzz }; return p; }
	//	static MaterialParams  MakeDielec(glm::vec3 albedo, float  ior) { MaterialParams p; p.matID = DIELECTRIC; p.dielec  = { albedo,  ior }; return p; }


	//	__device__ Material* MakeMaterial() const {
	//		switch (matID) {
	//		case LAMBERTIAN: return new LambertianAbstract(lambert.albedo);
	//		case      METAL: return new      MetalAbstract(metal.albedo, metal.fuzz);
	//		case DIELECTRIC: return new DielectricAbstract(dielec.albedo, dielec.ior);
	//		}
	//	}
	//};

	//class d_MaterialFactory {
	//	MaterialParams* p{};
	//public:
	//	__host__ __device__ d_MaterialFactory(MaterialParams* p) : p(p) {}
	//	__device__ Material* operator()(size_t index) const { return p[index].MakeMaterial(); }
	//};

	struct SphereParams {
		glm::vec3 origin;
		float radius;
		Material* mat_ptr;
		__device__ Hittable* MakeSphere() const { return new SphereHittable(origin, radius, mat_ptr); }
	};

	class d_SphereFactory {
		SphereParams* p{};
		//Material** mats;
	public:
		__host__ __device__ d_SphereFactory(SphereParams* p) : p(p) {}
		__device__ Hittable* operator()(size_t index) const { return p[index].MakeSphere(); }
	};

private:
	void _make_sphere(int a, int b) {
	#define rnd (host_rnd.next())

		float choose_mat = rnd;
		glm::vec3 pos(a + 0.9f * rnd, 0.2f, b + 0.9f * rnd);

		if (glm::length(pos - glm::vec3(4, 0.2f, 0)) > 0.9f) {
			if (choose_mat < 0.8f) {
				// diffuse
				glm::vec3 albedo = glm::vec3(rnd, rnd, rnd) * glm::vec3(rnd, rnd, rnd);
				
				auto material = dobj<LambertianAbstract>::Make(albedo);
				sphere_params.push_back({ pos, 0.2f, material.getPtr()});
				materials.push_back(std::move(material));
				//material_params.push_back(MaterialParams::MakeLambert(albedo));
			}
			else if (choose_mat < 0.95f) {
				// metal
				glm::vec3 albedo = glm::vec3(rnd, rnd, rnd) * 0.5f + 0.5f;
				float fuzz = rnd * 0.5f;

				auto material = dobj<MetalAbstract>::Make(albedo, fuzz);
				sphere_params.push_back({ pos, 0.2f, material.getPtr()});
				materials.push_back(std::move(material));
				//material_params.push_back(MaterialParams::MakeMetal(albedo, fuzz));
			}
			else {
				// glass
				auto material = dobj<DielectricAbstract>::Make(glm::vec3(1.0f), 1.5f);
				sphere_params.push_back({ pos, 0.2f, material.getPtr()});
				materials.push_back(std::move(material));
				//material_params.push_back(MaterialParams::MakeDielec(glm::vec3(1.0f), 1.5f));
			}
		}
	}

	void _populate_world() {
		// ground sphere
		auto ground_mat = dobj<LambertianAbstract>::Make(glm::vec3(0.5f));
		sphere_params.push_back({ glm::vec3(0,-1000,0), 1000, ground_mat.getPtr() });
		materials.push_back(std::move(ground_mat));
		//material_params.push_back(MaterialParams::MakeLambert(glm::vec3(0.5f)));

		for (int a = -11; a < 11; a++) {
			for (int b = -11; b < 11; b++) {
				_make_sphere(a, b);
			}
		}

		auto center_mat = dobj<DielectricAbstract>::Make(glm::vec3(1.0f), 1.5f);
		sphere_params.push_back({ glm::vec3(0,1,0),1, center_mat.getPtr()});
		materials.push_back(std::move(center_mat));
		//material_params.push_back(MaterialParams::MakeDielec(glm::vec3(1.0f), 1.5f));

		auto left_mat = dobj<LambertianAbstract>::Make(glm::vec3(0.4f, 0.2f, 0.1f));
		sphere_params.push_back({ glm::vec3(-4,1,0),1,left_mat.getPtr()});
		materials.push_back(std::move(left_mat));
		//material_params.push_back(MaterialParams::MakeLambert(glm::vec3(0.4f, 0.2f, 0.1f)));

		auto right_mat = dobj<MetalAbstract>::Make(glm::vec3(0.7f, 0.6f, 0.5f), 0);
		sphere_params.push_back({ glm::vec3(4,1,0),1,right_mat.getPtr()});
		materials.push_back(std::move(right_mat));
		//material_params.push_back(MaterialParams::MakeMetal(glm::vec3(0.7f, 0.6f, 0.5f), 0));
	}

	

	//std::vector<MaterialParams> material_params{};
	std::vector<dobj<Material>> materials;
	std::vector<  SphereParams> sphere_params;
	cuHostRND host_rnd{ 512, 1984 };

public:

	static _SceneDescription MakeScene() {
		SceneBook1FinaleFactory factory{};

		factory._populate_world();

		//darray<MaterialParams> d_mat_params(factory.material_params);
		//d_MaterialFactory material_factory(d_mat_params.getPtr());
		//auto d_mat_factory = dobj<d_MaterialFactory>::Make(material_factory);

		//dAbstractArray<Material> materials(factory.material_params.size());
		//materials.MakeOnDeviceFactory<d_MaterialFactory>(factory.material_params.size(), 0, 0, d_mat_factory.getPtr());


		darray<SphereParams> d_sphere_params(factory.sphere_params);
		d_SphereFactory sphere_factory(d_sphere_params.getPtr());
		auto d_sphere_factory = dobj<d_SphereFactory>::Make(sphere_factory);

		dAbstractArray<Hittable, true> sphere_list(factory.sphere_params.size());
		sphere_list.MakeOnDeviceFactory<d_SphereFactory>(factory.sphere_params.size(), 0, 0, d_sphere_factory.getPtr());

		
		auto world_list = dobj<HittableList>::Make(sphere_list.getDeviceArrayPtr(), sphere_list.getLength());

		return _SceneDescription {
			std::move(factory.materials),
			std::move(sphere_list),
			std::move(world_list)
		};
	}
};

#endif

FirstApp FirstApp::MakeApp() {
	uint32_t _width = 1280;
	uint32_t _height = 720;

	glm::vec3 lookfrom(13, 2, 3);
	glm::vec3 lookat(0, 0, 0);
	glm::vec3 up(0, 1, 0);
	float fov = 20.0f;
	float aspect = _width / (float)_height;
	PinholeCamera cam = PinholeCamera(lookfrom, lookat, up, fov, aspect);

	_SceneDescription scene_desc = SceneBook1FinaleFactory::MakeScene();

	Renderer renderer = Renderer::MakeRenderer(_width, _height, 8, 8, cam, scene_desc.world_list.getPtr());

	glm::vec4* host_output_framebuffer{};
	CUDA_ASSERT(cudaMallocHost(&host_output_framebuffer, sizeof(glm::vec4) * _width * _height));

	return FirstApp(M{
		_width,
		_height,
		cam,
		host_output_framebuffer,
		std::move(renderer),
		std::move(scene_desc),
	});
}
FirstApp::~FirstApp() {
	CUDA_ASSERT(cudaFreeHost(m.host_output_framebuffer));
}

void write_renderbuffer_png(std::string filepath, uint32_t width, uint32_t height, glm::vec4* data);
void FirstApp::Run() {
	m.renderer.Render();
	m.renderer.DownloadRenderbuffer(m.host_output_framebuffer);
	write_renderbuffer_png("../renders/test_040.png"s, m.render_width, m.render_height, m.host_output_framebuffer);
}

void write_renderbuffer_png(std::string filepath, uint32_t width, uint32_t height, glm::vec4* data) {
	//uint8_t* output_image_data = new uint8_t[width * height * 4];
	std::vector<uint8_t> output_image_data(width * height * 4);
	for (uint32_t i = 0; i < width * height; i++) {
		output_image_data.push_back(static_cast<uint8_t>(data[i][0] * 255.999f));
		output_image_data.push_back(static_cast<uint8_t>(data[i][1] * 255.999f));
		output_image_data.push_back(static_cast<uint8_t>(data[i][2] * 255.999f));
		output_image_data.push_back(static_cast<uint8_t>(data[i][3] * 255.999f));
	}

	stbi_flip_vertically_on_write(true);
	stbi_write_png(filepath.c_str(), width, height, 4, output_image_data.data(), sizeof(uint8_t) * width * 4);
	//delete[] output_image_data;
}