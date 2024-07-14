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

	Mesh();

	uint32_t vao, ebo, vbo;
	uint32_t indices;

public:

	~Mesh();
	Mesh(Mesh&& other);
	Mesh& operator=(Mesh&& other);

	void Draw() { Draw(0, indices); }
	void Draw(int offset, int elem_count);
	void Delete();

	static Mesh* LoadFromVertices(const std::vector<Vertex>& vertices);
	static Mesh* LoadFromVerticesAndIndices(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices);
};

} // gl_engine //

#endif // OPENGL_MESH_CLASS_H //