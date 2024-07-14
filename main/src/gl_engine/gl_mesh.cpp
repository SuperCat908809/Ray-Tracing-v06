#include "../pch.h"
#include "gl_mesh.h"

using namespace gl_engine;


void Mesh::_delete() {
	if (vao != 0)
		glDeleteVertexArrays(1, &vao);
	if (ebo != 0)
		glDeleteBuffers(1, &ebo);
	if (vbo != 0)
		glDeleteBuffers(1, &vbo);

	vao = 0;
	ebo = 0;
	vbo = 0;
}
Mesh::~Mesh() {
	_delete();
}
void Mesh::Delete() {
	_delete();
}
Mesh::Mesh(Mesh&& other) {
	indices = other.indices;

	vao = other.vao;
	ebo = other.vbo;
	vbo = other.vbo;

	other.vao = 0;
	other.vbo = 0;
	other.vbo = 0;
}
Mesh& Mesh::operator=(Mesh&& other) {
	_delete();

	indices = other.indices;

	vao = other.vao;
	ebo = other.vbo;
	vbo = other.vbo;

	other.vao = 0;
	other.vbo = 0;
	other.vbo = 0;

	return *this;
}


void Mesh::Draw(int offset, int elem_count) {
	glBindVertexArray(vao);
	if (ebo == 0) {
		glDrawArrays(GL_TRIANGLES, 0, indices);
	}
	else {
		glDrawElements(GL_TRIANGLES, elem_count, GL_UNSIGNED_INT, (void*)offset);
	}
}

Mesh* Mesh::LoadFromVerticesAndIndices(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices) {

	Mesh* mesh = new Mesh();

	glGenVertexArrays(1, &mesh->vao);
	glBindVertexArray(mesh->vao);

	glGenBuffers(1, &mesh->ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh->ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint32_t) * indices.size(), indices.data(), GL_STATIC_DRAW);
	mesh->indices = indices.size();

	glGenBuffers(1, &mesh->vbo);
	glBindBuffer(GL_ARRAY_BUFFER, mesh->vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * vertices.size(), vertices.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(0 * sizeof(float)));
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(3 * sizeof(float)));
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);

	glBindVertexArray(0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	return mesh;
}

Mesh* Mesh::LoadFromVertices(const std::vector<Vertex>& vertices) {

	Mesh* mesh = new Mesh();

	glGenVertexArrays(1, &mesh->vao);
	glBindVertexArray(mesh->vao);

	mesh->ebo = 0;
	mesh->indices = vertices.size();

	glGenBuffers(1, &mesh->vbo);
	glBindBuffer(GL_ARRAY_BUFFER, mesh->vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * vertices.size(), vertices.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(0 * sizeof(float)));
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(3 * sizeof(float)));
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);

	glBindVertexArray(0);

	return mesh;

}