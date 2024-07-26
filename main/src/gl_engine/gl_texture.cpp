#include "../pch.h"
#include "gl_texture.h"

#include <string>
#include <inttypes.h>

#include <glad/glad.h>
#include <stb/stb_image.h>
#include <stb/stb_image_resize.h>

using namespace gl_engine;


Texture::Texture() : width(0), height(0), internal_format(GL_FALSE) {}
void Texture::_delete() {
	if (id != 0)
		glDeleteTextures(1, &id);

	id = 0;
}
Texture::~Texture() {
	_delete();
}
void Texture::Delete() {
	_delete();
}
Texture::Texture(Texture&& other) noexcept {
	id = other.id;
	width = other.width;
	height = other.height;
	internal_format = other.internal_format;

	other.id = 0;
}
Texture& Texture::operator=(Texture&& other) noexcept {
	_delete();

	id = other.id;
	width = other.width;
	height = other.height;
	internal_format = other.internal_format;

	other.id = 0;

	return *this;
}

void Texture::Bind(int slot) {
	Texture::slot = slot;

	glActiveTexture(GL_TEXTURE0 + slot);
	glBindTexture(GL_TEXTURE_2D, id);
}

void Texture::SetWrappingMode(GLenum wrap_s, GLenum wrap_t) {
	Bind();
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_s);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap_t);
}
void Texture::SetInterpolationMode(GLenum min, GLenum mag) {
	Bind();
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, min);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mag);
}
void Texture::SetBorderColor(glm::vec4 color) {
	Bind();
	glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, glm::value_ptr(color));
}

void Texture::LoadTo(glm::uvec2 offset, glm::uvec2 size, int level, GLenum format, GLenum type, const void* pixels) {
	Bind();
	glTexSubImage2D(GL_TEXTURE_2D, level, offset.x, offset.y, size.x, size.y, format, type, pixels);
}

void Texture::ReadFrom(int src_level, GLenum dst_format, GLenum dst_type, void* dst_pixels) {
	Bind();
	glGetTexImage(GL_TEXTURE_2D, src_level, dst_format, dst_type, dst_pixels);
}

Texture* Texture::Make(uint32_t width, uint32_t height, GLenum internal_format, int slot) {
	Texture* tex = new Texture();

	tex->width = width;
	tex->height = height;
	tex->internal_format = internal_format;
	tex->slot = slot;

	glGenTextures(1, &tex->id);
	glActiveTexture(GL_TEXTURE0 + tex->slot);
	glBindTexture(GL_TEXTURE_2D, tex->id);
	glTexImage2D(GL_TEXTURE_2D, 0, tex->internal_format, tex->width, tex->height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

	tex->SetWrappingMode(GL_REPEAT, GL_REPEAT);
	tex->SetInterpolationMode(GL_NEAREST, GL_NEAREST);

	return tex;
}

Texture* Texture::LoadFromFile(std::string image_filename, int force_channels, glm::uvec2 force_resolution) noexcept(false) {
	if (force_channels < 0 || force_channels > 4) {
		throw std::runtime_error("force_channels " + std::to_string(force_channels) + " outside [0, 4] range");
	}

	int width, height, channels;
	stbi_set_flip_vertically_on_load(true);
	uint8_t* image_data = stbi_load(image_filename.c_str(), &width, &height, &channels, force_channels);

	if (image_data == nullptr) {
		throw std::runtime_error(image_filename + " not found");
	}

	if (force_channels == 0) {
		force_channels = channels;
	}

	GLenum internal_format = GL_R8;
	GLenum image_format = GL_RED;
	switch (force_channels) {
	case 1: internal_format = GL_R8;    image_format = GL_RED; break;
	case 2: internal_format = GL_RG8;   image_format = GL_RG; break;
	case 3: internal_format = GL_RGB8;  image_format = GL_RGB; break;
	case 4: internal_format = GL_RGBA8; image_format = GL_RGBA; break;
	}

	Texture* tex{ nullptr };
	if (force_resolution == glm::uvec2(0, 0) || force_resolution == glm::uvec2(width, height)) {
		tex = Make(width, height, internal_format);
		tex->LoadTo({ 0,0 }, { width,height }, 0, image_format, GL_UNSIGNED_BYTE, image_data);
	}
	else {
		uint8_t* resize = new uint8_t[force_channels * force_resolution.x * force_resolution.y];

		stbir_resize_uint8(image_data, width, height, 0, resize, force_resolution.x, force_resolution.y, 0, force_channels);

		tex = Make(force_resolution.x, force_resolution.y, internal_format);
		tex->LoadTo({ 0,0 }, force_resolution, 0, image_format, GL_UNSIGNED_BYTE, resize);

		delete[] resize;
	}

	stbi_image_free(image_data);

	return tex;
}

#if 0
Texture* Texture::LoadFromImageFileRGB(std::string image_filename) {

	int width, height, channels;
	stbi_set_flip_vertically_on_load(true);
	uint8_t* image_data = stbi_load(image_filename.c_str(), &width, &height, &channels, 3);

	Texture* tex = LoadFromPixelDataRGB(width, height, image_data);

	stbi_image_free(image_data);

	return tex;
}
Texture* Texture::LoadFromImageFileRGBA(std::string image_filename) {

	int width, height, channels;
	stbi_set_flip_vertically_on_load(true);
	uint8_t* image_data = stbi_load(image_filename.c_str(), &width, &height, &channels, 4);

	Texture* tex = LoadFromPixelDataRGBA(width, height, image_data);

	stbi_image_free(image_data);

	return tex;
}
Texture* Texture::LoadFromPixelDataRGB(uint32_t width, uint32_t height, const uint8_t* data) {

	Texture* tex = new Texture();

	glGenTextures(1, &tex->id);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tex->id);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
	glBindTexture(GL_TEXTURE_2D, 0);

	return tex;
}
Texture* Texture::LoadFromPixelDataRGBA(uint32_t width, uint32_t height, const uint8_t* data) {

	Texture* tex = new Texture();

	glGenTextures(1, &tex->id);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tex->id);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
	glBindTexture(GL_TEXTURE_2D, 0);

	return tex;
}
#endif