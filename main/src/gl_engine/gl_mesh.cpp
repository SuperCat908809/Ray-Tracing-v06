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
	glDrawElements(GL_TRIANGLES, elem_count, GL_UNSIGNED_INT, (void*)offset);
}

Mesh::Mesh(std::vector<Vertex> vertices, std::vector<uint32_t> indices) : indices(indices.size()) {
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint32_t) * indices.size(), indices.data(), GL_STATIC_DRAW);

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * vertices.size(), vertices.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(0 * sizeof(float)));
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(3 * sizeof(float)));
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);

	glBindVertexArray(0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}