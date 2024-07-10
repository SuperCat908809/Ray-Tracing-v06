#include "../pch.h"
#include "shader.h"

#include <errno.h>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>

#include <glad/glad.h>


void Shader::_delete() {
	if (id != 0)
		glDeleteProgram(id);

	id = 0;
}
Shader::~Shader() {
	_delete();
}

Shader::Shader(Shader&& other) {
	id = other.id;

	other.id = 0;
}
Shader& Shader::operator=(Shader&& other) {
	_delete();

	id = other.id;

	other.id = 0;

	return *this;
}

std::string get_file_contents(std::string filename) {
	std::ifstream in(filename, std::ios::binary);
	if (in) {
		std::string contents{};
		in.seekg(0, std::ios::end);
		contents.resize(in.tellg());
		in.seekg(0, std::ios::beg);
		in.read(&contents[0], contents.size());
		in.close();
		return (contents);
	}
	throw (errno);
}

Shader::Shader(std::string vertex_file, std::string fragment_file) {

	std::string vert_shader_source = get_file_contents(vertex_file);
	const char* vert_ptr = vert_shader_source.c_str();

	std::string frag_shader_source = get_file_contents(fragment_file);
	const char* frag_ptr = frag_shader_source.c_str();

	uint32_t vert_shader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vert_shader, 1, &vert_ptr, nullptr);
	glCompileShader(vert_shader);

	uint32_t frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(frag_shader, 1, &frag_ptr, nullptr);
	glCompileShader(frag_shader);

	id = glCreateProgram();
	glAttachShader(id, vert_shader);
	glAttachShader(id, frag_shader);
	glLinkProgram(id);

	glDetachShader(id, vert_shader);
	glDetachShader(id, frag_shader);
	glDeleteShader(vert_shader);
	glDeleteShader(frag_shader);

}

void Shader::Use() {
	glUseProgram(id);
}
void Shader::Delete() {
	_delete();
}