#ifndef OPENGL_TEXTURE_CLASS_H
#define OPENGL_TEXTURE_CLASS_H

#include <inttypes.h>
#include <string>


namespace gl_engine {
class Texture{

	Texture(Texture&) = delete;
	Texture& operator=(Texture&) = delete;

	Texture();

	void _delete();

public:

	~Texture();
	Texture(Texture&& other);
	Texture& operator=(Texture&& other);

	uint32_t id = 0;
	uint32_t slot = 0;

	void Bind(int slot);
	void Bind() { Bind(slot); }

	void Delete();

	static Texture* LoadFromImageFileRGB(std::string filepath);
	static Texture* LoadFromImageFileRGBA(std::string filepath);
	static Texture* LoadFromPixelDataRGB(uint32_t width, uint32_t height, const uint8_t* data);
	static Texture* LoadFromPixelDataRGBA(uint32_t width, uint32_t height, const uint8_t* data);
};
}

#endif // OPENGL_TEXTURE_CLASS_H //