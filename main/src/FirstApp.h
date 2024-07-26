#ifndef FIRST_APP_CLASS_H
#define FIRST_APP_CLASS_H

#include <inttypes.h>
#include <glm/glm.hpp>


class Renderer;
class MotionBlurCamera;
class SceneBook2BVH;

class FirstApp {

	struct M {
		uint32_t render_width{}, render_height{};
		MotionBlurCamera* cam{};
		glm::vec4* host_output_framebuffer{};
		Renderer* renderer;

		SceneBook2BVH* _sceneDesc;
	} m;

	FirstApp(M m) : m(std::move(m)) {}

	FirstApp(const FirstApp&) = delete;
	FirstApp& operator=(const FirstApp&) = delete;

	void _delete();

public:

	static FirstApp MakeApp();
	FirstApp(FirstApp&& other);
	FirstApp& operator=(FirstApp&& other);
	~FirstApp();

	void Run();
};

#endif // FIRST_APP_CLASS_H //