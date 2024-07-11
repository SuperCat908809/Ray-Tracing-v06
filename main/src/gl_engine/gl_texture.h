#ifndef OPENGL_TEXTURE_CLASS_H
#define OPENGL_TEXTURE_CLASS_H

#include <inttypes.h>
#include <string>


namespace gl_engine {
class Texture{

	Texture(Texture&) = delete;
	Texture& operator=(Texture&) = delete;

	void _delete();

public:

	~Texture();
	Texture(Texture&& other);
	Texture& operator=(Texture&& other);

	uint32_t id;
	uint32_t slot = 0;

	Texture(std::string image_filename);

	void Bind(int slot);
	void Bind() { Bind(slot); }

	void Delete();
};
}

#endif // OPENGL_TEXTURE_CLASS_H //