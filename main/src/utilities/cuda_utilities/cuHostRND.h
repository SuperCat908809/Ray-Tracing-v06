#ifndef CU_HOST_RND_CLASSES_H
#define CU_HOST_RND_CLASSES_H

#include <inttypes.h>
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

	cuHostRND(cuHostRND&& other) noexcept;
	cuHostRND& operator=(cuHostRND&& other) noexcept;

	cuHostRND(size_t capacity, size_t seed, size_t offset = 0, curandOrdering_t ordering = CURAND_ORDERING_PSEUDO_DEFAULT);
	~cuHostRND();

	float next();
};

#endif // CU_HOST_RND_CLASSES_H //