#ifndef OPENGL_MESH_CLASS_H
#define OPENGL_MESH_CLASS_H

#include <inttypes.h>
#include <glm/glm.hpp>
#include <vector>


namespace gl_engine {

struct Vertex{
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 tex_coord;
};

class Mesh {

	Mesh(Mesh&) = delete;
	Mesh& operator=(Mesh&) = delete;

	void _delete();

	uint32_t vao, ebo, vbo;
	uint32_t indices;
public:

	Mesh(Mesh&& other);
	Mesh& operator=(Mesh&& other);
	~Mesh();

	Mesh(std::vector<Vertex> vertices, std::vector<uint32_t> indices);

	void Draw() { Draw(0, indices); }
	void Draw(int offset, int elem_count);
	void Delete();

};

} // gl_engine //

#endif // OPENGL_MESH_CLASS_H //