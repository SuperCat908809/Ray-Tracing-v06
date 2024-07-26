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
	Texture(Texture&& other) noexcept;
	Texture& operator=(Texture&& other) noexcept;

	uint32_t id = 0;
	uint32_t slot = 0;
	uint32_t width, height;
	GLenum internal_format;

	void Bind(int slot);
	void Bind() { Bind(slot); }

	void Delete();

	void SetWrappingMode(GLenum wrap_s, GLenum wrap_t);
	void SetInterpolationMode(GLenum min, GLenum mag);
	void SetBorderColor(glm::vec4 color);

	void LoadTo(glm::uvec2 offset, glm::uvec2 size, int dst_level, GLenum src_format, GLenum src_type, const void* pixels);
	void ReadFrom(int src_level, GLenum dst_format, GLenum dst_type, void* dst_pixels);

	static Texture* Make(uint32_t width, uint32_t height, GLenum internal_format, int slot = 0);

	static Texture* LoadFromFile(std::string image_filename, int force_channels = 0, glm::uvec2 force_resolution = { 0,0 }) noexcept(false);
	//static Texture* LoadFromPixelDataRGB(uint32_t width, uint32_t height, const uint8_t* data);
	//static Texture* LoadFromPixelDataRGBA(uint32_t width, uint32_t height, const uint8_t* data);
};
}

#endif // OPENGL_TEXTURE_CLASS_H //