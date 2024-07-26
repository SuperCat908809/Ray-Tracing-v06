#include "../pch.h"
#include "gl_renderbuffer.h"

#include "assert.h"

using namespace gl_engine;


#if 0
template <GL_COMPONENT, GL_SIZE, GL_TYPE> GLenum getFormat() { assert(false, "Invalid renderbuffer configuration"); return GL_FALSE; }

template <> GLenum getFormat<RED, QUARTER, NORM>() { return GL_R8; }
template <> GLenum getFormat<RG, QUARTER, NORM>() { return GL_RG8; }
template <> GLenum getFormat<RGBA, QUARTER, NORM>() { return GL_RGBA8; }

template <> GLenum getFormat<RED, HALF, NORM>() { return GL_R16; }
template <> GLenum getFormat<RG, HALF, NORM>() { return GL_RG16; }
template <> GLenum getFormat<RGBA, HALF, NORM>() { return GL_RGBA16; }

template <> GLenum getFormat<RED, QUARTER, SNORM>() { return GL_R8_SNORM; }
template <> GLenum getFormat<RG, QUARTER, SNORM>() { return GL_RG8_SNORM; }
template <> GLenum getFormat<RGBA, QUARTER, SNORM>() { return GL_RGBA8_SNORM; }

template <> GLenum getFormat<RED, HALF, SNORM>() { return GL_R16_SNORM; }
template <> GLenum getFormat<RG, HALF, SNORM>() { return GL_RG16_SNORM; }
template <> GLenum getFormat<RGBA, HALF, SNORM>() { return GL_RGBA16_SNORM; }


template <> GLenum getFormat<RED, QUARTER, UINT>() { return GL_R8UI; }
template <> GLenum getFormat<RG, QUARTER, UINT>() { return GL_RG8UI; }
template <> GLenum getFormat<RGBA, QUARTER, UINT>() { return GL_RGBA8UI; }

template <> GLenum getFormat<RED, HALF, UINT>() { return GL_R16UI; }
template <> GLenum getFormat<RG, HALF, UINT>() { return GL_RG16UI; }
template <> GLenum getFormat<RGBA, HALF, UINT>() { return GL_RGBA16UI; }

template <> GLenum getFormat<RED, FULL, UINT>() { return GL_R32UI; }
template <> GLenum getFormat<RG, FULL, UINT>() { return GL_RG32UI; }
template <> GLenum getFormat<RGBA, FULL, UINT>() { return GL_RGBA32UI; }

template <> GLenum getFormat<RED, QUARTER, INT>() { return GL_R8I; }
template <> GLenum getFormat<RG, QUARTER, INT>() { return GL_RG8I; }
template <> GLenum getFormat<RGBA, QUARTER, INT>() { return GL_RGBA8I; }

template <> GLenum getFormat<RED, HALF, INT>() { return GL_R16I; }
template <> GLenum getFormat<RG, HALF, INT>() { return GL_RG16I; }
template <> GLenum getFormat<RGBA, HALF, INT>() { return GL_RGBA16I; }

template <> GLenum getFormat<RED, FULL, INT>() { return GL_R32I; }
template <> GLenum getFormat<RG, FULL, INT>() { return GL_RG32I; }
template <> GLenum getFormat<RGBA, FULL, INT>() { return GL_RGBA32I; }


template <> GLenum getFormat<RED, HALF, FLOAT>() { return GL_R16F; }
template <> GLenum getFormat<RG, HALF, FLOAT>() { return GL_RG16F; }
template <> GLenum getFormat<RGBA, HALF, FLOAT>() { return GL_RGBA16F; }

template <> GLenum getFormat<RED, FULL, FLOAT>() { return GL_R32F; }
template <> GLenum getFormat<RG, FULL, FLOAT>() { return GL_RG32F; }
template <> GLenum getFormat<RGBA, FULL, FLOAT>() { return GL_RGBA32F; }
#endif



Renderbuffer::Renderbuffer() : id(0), width(0), height(0), internal_format(GL_FALSE) {}
Renderbuffer::~Renderbuffer() {
	_delete();
}
void Renderbuffer::Delete() {
	_delete();
}
void Renderbuffer::_delete() {
	if (id != 0)
		glDeleteRenderbuffers(1, &id);
	id = 0;
}

Renderbuffer::Renderbuffer(Renderbuffer&& other) noexcept {
	id = other.id;
	width = other.width;
	height = other.height;
	internal_format = other.internal_format;

	other.id = 0;
}
Renderbuffer& Renderbuffer::operator=(Renderbuffer&& other) noexcept {
	_delete();

	id = other.id;
	width = other.width;
	height = other.height;
	internal_format = other.internal_format;

	other.id = 0;

	return *this;
}

void Renderbuffer::Bind() {
	glBindRenderbuffer(GL_RENDERBUFFER, id);
}

Renderbuffer* Renderbuffer::Make(uint32_t width, uint32_t height, GLenum format) {
	Renderbuffer* rb = new Renderbuffer();

	rb->width = width;
	rb->height = height;
	rb->internal_format = format;

	glGenRenderbuffers(1, &rb->id);
	glBindRenderbuffer(GL_RENDERBUFFER, rb->id);
	glRenderbufferStorage(GL_RENDERBUFFER, format, width, height);

	return rb;
}