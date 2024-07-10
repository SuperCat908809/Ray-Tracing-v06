#ifndef SHADER_CLASS_H
#define SHADER_CLASS_H

#include <inttypes.h>
#include <string>


class Shader {

	Shader(Shader&) = delete;
	Shader& operator=(Shader&) = delete;

	void _delete();

public:

	uint32_t id;
	Shader(std::string vertex_file, std::string fragment_file);
	~Shader();
	Shader(Shader&& other);
	Shader& operator=(Shader&& other);

	void Use();
	void Delete();

};

#endif // SHADER_CLASS_H //