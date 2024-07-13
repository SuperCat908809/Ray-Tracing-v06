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


std::string get_file_contents(std::string filename);
void compileErrors(uint32_t shader, std::string filename);
void linkErrors(uint32_t program);

Shader::Shader(std::string vertex_file, std::string fragment_file) {

	std::string vert_shader_source = get_file_contents(vertex_file);
	const char* vert_ptr = vert_shader_source.c_str();

	std::string frag_shader_source = get_file_contents(fragment_file);
	const char* frag_ptr = frag_shader_source.c_str();

	uint32_t vert_shader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vert_shader, 1, &vert_ptr, nullptr);
	glCompileShader(vert_shader);
	compileErrors(vert_shader, vertex_file);

	uint32_t frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(frag_shader, 1, &frag_ptr, nullptr);
	glCompileShader(frag_shader);
	compileErrors(frag_shader, fragment_file);

	id = glCreateProgram();
	glAttachShader(id, vert_shader);
	glAttachShader(id, frag_shader);
	glLinkProgram(id);
	linkErrors(id);

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

void compileErrors(uint32_t shader, std::string filename) {
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