#include "../pch.h"
#include "gl_mesh.h"

#include <tuple>
#include <map>

#include <tiny_obj/tiny_obj_loader.h>

using namespace gl_engine;


Mesh::Mesh() : vao(0), ebo(0), vbo(0), indices(0) {}
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
Mesh::Mesh(Mesh&& other) noexcept {
	indices = other.indices;

	vao = other.vao;
	ebo = other.vbo;
	vbo = other.vbo;

	other.vao = 0;
	other.vbo = 0;
	other.vbo = 0;
}
Mesh& Mesh::operator=(Mesh&& other) noexcept {
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
		glDrawArrays(GL_TRIANGLES, offset, elem_count);
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


Mesh* Mesh::LoadFromObjFile(const std::string& filepath) noexcept(false) {

	tinyobj::ObjReaderConfig reader_config{};
	reader_config.mtl_search_path = "";
	reader_config.triangulate = true;
	reader_config.vertex_color = false;

	tinyobj::ObjReader reader;

	if (!reader.ParseFromFile(filepath, reader_config)) {
		if (!reader.Error().empty()) {
			throw std::runtime_error("TinyObjReader: " + reader.Error());
		}
		throw std::runtime_error("TinyObjReader: unspecified error");
	}

	if (!reader.Warning().empty()) {
		std::cerr << "TinyObjReader: " << reader.Warning();
	}

	auto& attrib = reader.GetAttrib();
	auto& shapes = reader.GetShapes();
	auto& materials = reader.GetMaterials();

	std::vector<Vertex> vertices{};
	std::vector<uint32_t> indices{};
	std::map<std::tuple<int, int, int>, uint32_t> index_map{};

	// Loop over shapes
	for (size_t s = 0; s < shapes.size(); s++) {
		// Loop over faces(polygon)
		size_t index_offset = 0;
		for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
			size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

			// Loop over vertices in the face.
			for (size_t v = 0; v < fv; v++) {
				// access to vertex
				tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
				auto map_idx = std::make_tuple(idx.vertex_index, idx.normal_index, idx.texcoord_index);

				auto map_index = index_map.find(map_idx);
				if (map_index != index_map.end()) {
					indices.push_back(map_index->second);
					continue;
				}

				Vertex vertex{};

				tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
				tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
				tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

				vertex.position = glm::vec3(vx, vy, vz);

				// Check if `normal_index` is zero or positive. negative = no normal data
				if (idx.normal_index >= 0) {
					tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
					tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
					tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
					vertex.normal = glm::vec3(nx, ny, nz);
				}
				else {
					vertex.normal = glm::vec3(0, 1, 0);
				}

				// Check if `texcoord_index` is zero or positive. negative = no texcoord data
				if (idx.texcoord_index >= 0) {
					tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
					tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
					vertex.tex_coord = glm::vec2(tx, ty);
				}
				else {
					vertex.tex_coord = glm::vec2(0, 0);
				}

				// Optional: vertex colors
				// tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
				// tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
				// tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];

				int vert_idx = vertices.size();
				vertices.push_back(vertex);
				indices.push_back(vert_idx);
				index_map.insert(std::make_pair(map_idx, vert_idx));
			}
			index_offset += fv;

			// per-face material
			//shapes[s].mesh.material_ids[f];
		}
	}

	return LoadFromVerticesAndIndices(vertices, indices);

}