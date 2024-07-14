#include "../pch.h"
#include "gl_shader.h"

#include <stdexcept>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>

#include <glad/glad.h>


Shader::Shader() = default;
void Shader::_delete() {
	if (id != 0)
		glDeleteProgram(id);

	id = 0;
}
Shader::~Shader() {
	_delete();
}

Shader::Shader(Shader&& other) noexcept {
	id = other.id;

	other.id = 0;
}
Shader& Shader::operator=(Shader&& other) noexcept {
	_delete();

	id = other.id;

	other.id = 0;

	return *this;
}


std::string get_file_contents(std::string filename) noexcept(false);
void compileErrors(uint32_t shader) noexcept(false);
void linkErrors(uint32_t program) noexcept(false);

Shader* Shader::LoadFromFiles(std::string vertex_path, std::string fragment_path) noexcept(false) {
	return LoadFromSource(get_file_contents(vertex_path), get_file_contents(fragment_path));
}

Shader* Shader::LoadFromSource(const std::string& vertex_source, const std::string& fragment_source) noexcept(false) {

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


std::string get_file_contents(std::string filename) noexcept(false) {
	std::ifstream in(filename, std::ios::binary);
	if (!in) throw std::runtime_error("file " + filename + " not found");
	std::string contents{};
	in.seekg(0, std::ios::end);
	contents.resize(in.tellg());
	in.seekg(0, std::ios::beg);
	in.read(&contents[0], contents.size());
	in.close();
	return (contents);
}

void compileErrors(uint32_t shader) noexcept(false) {
	int has_compiled;
	std::string info_log{};

	glGetShaderiv(shader, GL_COMPILE_STATUS, &has_compiled);
	if (has_compiled == GL_FALSE) {
		int length;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
		info_log.resize(length);
		glGetShaderInfoLog(shader, length, nullptr, info_log.data());
		throw std::runtime_error("Shader compilation error:\n" + info_log);
	}
}

void linkErrors(uint32_t program) noexcept(false) {
	int has_compiled;
	std::string info_log{};

	glGetProgramiv(program, GL_COMPILE_STATUS, &has_compiled);
	if (has_compiled == GL_FALSE) {
		int length;
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
		info_log.resize(length);
		glGetProgramInfoLog(program, length, nullptr, info_log.data());
		throw std::runtime_error("Shader linking error:\n" + info_log);
	}
}