#ifndef SHADER_CLASS_H
#define SHADER_CLASS_H

#include <inttypes.h>
#include <string>
#include <stdexcept>


class Shader {

	Shader(Shader&) = delete;
	Shader& operator=(Shader&) = delete;

	Shader() = default;

	void _delete();

public:

	uint32_t id;
	~Shader();
	Shader(Shader&& other) noexcept;
	Shader& operator=(Shader&& other) noexcept;

	void Use();
	void Delete();

	static Shader* LoadFromFiles(std::string vertex_file, std::string fragment_file) noexcept(false);
	static Shader* LoadFromSource(const std::string& vertex_source, const std::string& fragment_source) noexcept(false);

};

#endif // SHADER_CLASS_H //