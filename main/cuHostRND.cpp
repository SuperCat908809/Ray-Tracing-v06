#include "cuHostRND.h"

#include <cuda_runtime.h>
#include <curand.h>
#include "cuError.h"
#include "darray.cuh"


void cuHostRND::_populate_buffer() {
	darray<float> d_rnd_uniforms(rnd_uniforms.size());

	curandGenerateUniform(gen, d_rnd_uniforms.getPtr(), rnd_uniforms.size());
	CUDA_ASSERT(cudaMemcpy(rnd_uniforms.data(), d_rnd_uniforms.getPtr(), sizeof(float) * rnd_uniforms.size(), cudaMemcpyDeviceToHost));
}

void cuHostRND::_delete() {
	if (gen) {
		curandDestroyGenerator(gen);
		gen = nullptr;
	}
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

cuHostRND::cuHostRND(size_t capacity, size_t seed, size_t offset, curandOrdering_t ordering) : rnd_uniforms(capacity), head(0ull) {
	rnd_uniforms.resize(capacity);

	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);
	curandSetPseudoRandomGeneratorSeed(gen, seed);
	curandSetGeneratorOffset(gen, offset);
	curandSetGeneratorOrdering(gen, ordering);

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