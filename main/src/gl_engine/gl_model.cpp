#include "../pch.h"
#include "gl_model.h"

#include <glad/glad.h>

using namespace gl_engine;

Model::Model() = default;

void Model::Draw() const {
	shader->Use();

	for (int i = 0; i < textures.size(); i++) {

		std::string name = std::get<0>(textures[i]);
		const std::shared_ptr<Texture>& tex = std::get<1>(textures[i]);

		int uni_loc = glGetUniformLocation(shader->id, name.c_str());
		glUniform1i(uni_loc, tex->slot);

		tex->Bind();
	}

	mesh->Draw();
}

Model* Model::MakeFrom(const std::shared_ptr<Mesh>& mesh, const std::shared_ptr<Shader>& shader,
	const std::vector<std::tuple<std::string, std::shared_ptr<Texture>>>& textures) {

	Model* model = new Model();

	model->mesh = mesh;
	model->shader = shader;
	model->textures = textures;

	return model;

}