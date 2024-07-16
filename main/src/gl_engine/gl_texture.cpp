#include "../pch.h"
#include "gl_texture.h"

#include <string>
#include <inttypes.h>

#include <glad/glad.h>
#include <stb/stb_image.h>

using namespace gl_engine;


Texture::Texture() : width(0), height(0) {}
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

	other.id = 0;
}
Texture& Texture::operator=(Texture&& other) noexcept {
	_delete();

	id = other.id;
	width = other.width;
	height = other.height;

	other.id = 0;

	return *this;
}

void Texture::Bind(int slot) {
	Texture::slot = slot;

	glActiveTexture(GL_TEXTURE0 + slot);
	glBindTexture(GL_TEXTURE_2D, id);
}

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