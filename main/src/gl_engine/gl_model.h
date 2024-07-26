#ifndef OPENGL_MODEL_CLASS_H
#define OPENGL_MODEL_CLASS_H

#include <vector>
#include <tuple>
#include <memory>
#include <string>

#include "gl_mesh.h"
#include "gl_shader.h"
#include "gl_texture.h"


namespace gl_engine {
class Model{

	Model();

public:

	std::shared_ptr<Mesh> mesh;
	std::shared_ptr<Shader> shader;
	std::vector<std::tuple<std::string, std::shared_ptr<Texture>>> textures;

	void Draw() const;

	static Model* MakeFrom(const std::shared_ptr<Mesh>& mesh, const std::shared_ptr<Shader>& shader,
		const std::vector<std::tuple<std::string, std::shared_ptr<Texture>>>& textures);
};
}

#endif // OPENGL_MODEL_CLASS_H //