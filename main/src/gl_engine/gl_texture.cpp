#include "../pch.h"
#include "gl_texture.h"

#include <string>
#include <inttypes.h>

#include <glad/glad.h>
#include <stb/stb_image.h>

using namespace gl_engine;


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
Texture::Texture(Texture&& other) {
	id = other.id;

	other.id = 0;
}
Texture& Texture::operator=(Texture&& other) {
	_delete();

	id = other.id;

	other.id = 0;

	return *this;
}

void Texture::Bind(int slot) {
	Texture::slot = slot;

	glActiveTexture(GL_TEXTURE0 + slot);
	glBindTexture(GL_TEXTURE_2D, id);
}

Texture::Texture(std::string image_filename) {

	int width, height, channels;
	stbi_set_flip_vertically_on_load(true);
	uint8_t* image_data = stbi_load(image_filename.c_str(), &width, &height, &channels, 3);

	glGenTextures(1, &id);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, id);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data);

	stbi_image_free(image_data);
	glBindTexture(GL_TEXTURE_2D, 0);

}