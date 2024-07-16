#ifndef OPENGL_RENDER_BUFFER_CLASS_H
#define OPENGL_RENDER_BUFFER_CLASS_H

#include <inttypes.h>

namespace gl_engine {

enum GL_COMPONENT { RED, RG, RGBA };
enum GL_TYPE { NORM, SNORM, UINT, INT, FLOAT };
enum GL_SIZE { QUARTER, HALF, FULL };
template <GL_COMPONENT, GL_SIZE, GL_TYPE> GLenum getFormat();

class Renderbuffer{

	Renderbuffer(Renderbuffer&) = delete;
	Renderbuffer& operator=(Renderbuffer&) = delete;

	Renderbuffer();

	void _delete();

public:

	uint32_t id, width, height;

	~Renderbuffer();
	Renderbuffer(Renderbuffer&& other) noexcept;
	Renderbuffer& operator=(Renderbuffer&& other) noexcept;

	void Bind();
	void Delete();

	static Renderbuffer* Make(uint32_t width, uint32_t height, GLenum format);
};
} // gl_engine //

#endif // OPENGL_RENDER_BUFFER_CLASS_H //