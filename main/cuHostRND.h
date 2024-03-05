#ifndef CU_HOST_RND_CLASSES_H
#define CU_HOST_RND_CLASSES_H

#include <cuda_runtime.h>
#include <curand.h>
#include <vector>

class cuHostRND {

	enum { MULTIPLIER_FACTOR = 2};

	std::vector<float> rnd_uniforms{};
	size_t head{ 0ull };
	curandGenerator_t gen;

	void _populate_buffer();
	void _delete();

public:

	cuHostRND(const cuHostRND&) = delete;
	cuHostRND& operator=(const cuHostRND&) = delete;

	cuHostRND(cuHostRND&& other);
	cuHostRND& operator=(cuHostRND&& other);

	cuHostRND(size_t capacity, size_t seed);
	~cuHostRND();

	float next();
};

#endif // CU_HOST_RND_CLASSES_H //