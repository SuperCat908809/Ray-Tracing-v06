#include "cuHostRND.h"

#include <cuda_runtime.h>
#include <curand.h>
#include "cuError.h"


void cuHostRND::_populate_buffer() {
	float* d_rnd_uniforms{ nullptr };
	CUDA_ASSERT(cudaMalloc((void**)&d_rnd_uniforms, sizeof(float) * rnd_uniforms.size()));

	curandGenerateUniform(gen, d_rnd_uniforms, rnd_uniforms.size());

	CUDA_ASSERT(cudaMemcpy(rnd_uniforms.data(), d_rnd_uniforms, sizeof(float) * rnd_uniforms.size(), cudaMemcpyDeviceToHost));

	CUDA_ASSERT(cudaFree(d_rnd_uniforms));
}

void cuHostRND::_delete() {
	curandDestroyGenerator(gen);
}

cuHostRND::cuHostRND(cuHostRND&& other) noexcept :
	rnd_uniforms(std::move(other.rnd_uniforms)),
	head(other.head),
	gen(other.gen) {
	other.gen = nullptr;
}

cuHostRND& cuHostRND::operator=(cuHostRND&& other) noexcept {
	_delete();

	rnd_uniforms = std::move(other.rnd_uniforms);
	head = other.head;
	gen = other.gen;

	other.gen = nullptr;

	return *this;
}

cuHostRND::cuHostRND(size_t capacity, size_t seed) {
	rnd_uniforms.resize(capacity);

	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);
	curandSetPseudoRandomGeneratorSeed(gen, seed);

	_populate_buffer();
}

cuHostRND::~cuHostRND() {
	_delete();
}

float cuHostRND::next() {
	if (head == rnd_uniforms.size()) {
		rnd_uniforms.resize(rnd_uniforms.size() * MULTIPLIER_FACTOR);
		head = 0ull;
		_populate_buffer();
	}

	return rnd_uniforms[head++];
}