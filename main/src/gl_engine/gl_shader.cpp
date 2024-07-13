#include "../pch.h"
#include "gl_shader.h"

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


std::string get_file_contents(std::string filename);
void compileErrors(uint32_t shader);
void linkErrors(uint32_t program);

Shader* Shader::LoadFromFiles(std::string vertex_path, std::string fragment_path) {
	return LoadFromSource(get_file_contents(vertex_path), get_file_contents(fragment_path));
}

Shader* Shader::LoadFromSource(const std::string& vertex_source, const std::string& fragment_source) {

	Shader* shader = new Shader();

	const char* vert_ptr = vertex_source.c_str();
	const char* frag_ptr = fragment_source.c_str();

	uint32_t vert_shader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vert_shader, 1, &vert_ptr, nullptr);
	glCompileShader(vert_shader);
	compileErrors(vert_shader);

	uint32_t frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(frag_shader, 1, &frag_ptr, nullptr);
	glCompileShader(frag_shader);
	compileErrors(frag_shader);

	shader->id = glCreateProgram();
	glAttachShader(shader->id, vert_shader);
	glAttachShader(shader->id, frag_shader);
	glLinkProgram(shader->id);
	linkErrors(shader->id);

	glDetachShader(shader->id, vert_shader);
	glDetachShader(shader->id, frag_shader);
	glDeleteShader(vert_shader);
	glDeleteShader(frag_shader);

	return shader;
}

void Shader::Use() {
	glUseProgram(id);
}
void Shader::Delete() {
	_delete();
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

void compileErrors(uint32_t shader) {
	int has_compiled;
	char infoLog[1024];
	glGetShaderiv(shader, GL_COMPILE_STATUS, &has_compiled);
	if (has_compiled == GL_FALSE) {
		glGetShaderInfoLog(shader, 1024, nullptr, infoLog);
		std::cerr << "Shader compilation error:\n" << infoLog;
		assert(has_compiled);
	}
}

void linkErrors(uint32_t program) {
	int has_compiled;
	char infoLog[1024];
	glGetProgramiv(program, GL_COMPILE_STATUS, &has_compiled);
	if (has_compiled == GL_FALSE) {
		glGetProgramInfoLog(program, 1024, nullptr, infoLog);
		std::cerr << "Program linking error:\n" << infoLog;
		assert(has_compiled);
	}
}