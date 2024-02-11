#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_utils.h"

#include <glm/glm.hpp>


int main() {

	glm::vec3 a(0, 0, 1);
	glm::vec3 b(1, 0, 0);
	glm::vec3 c = glm::cross(a, b);

	printf("c %.3f %.3f %.3f", c.x, c.y, c.z);

	std::cout << "\n\nFinished\n.";
	return 0;
}