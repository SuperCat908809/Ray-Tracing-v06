#ifndef OPENGL_FRAME_BUFFER_CLASS_H
#define OPENGL_FRAME_BUFFER_CLASS_H

#include <inttypes.h>
#include <glm/glm.hpp>


namespace gl_engine {

class Texture;
class Renderbuffer;

class Framebuffer{

	Framebuffer(Framebuffer&) = delete;
	Framebuffer& operator=(Framebuffer&) = delete;

	Framebuffer();

	void _delete();

public:

	uint32_t id, width, height;

	~Framebuffer();
	Framebuffer(Framebuffer&& other) noexcept;
	Framebuffer& operator=(Framebuffer&& other) noexcept;

	static Framebuffer* Make(uint32_t width, uint32_t height);

	//enum GL_FB_BIND_STATE { BOTH, DRAW, READ};
	//enum GL_FB_BUFFER { COLOR = 0b001, DEPTH = 0b010, STENCIL = 0b100 };
	//enum GL_FB_INTERPOLATION { NEAREST, LINEAR };

	void Bind(GLenum binder);
	static void ClearBinding();

	void BindTexture(Texture& tex, GLenum buffer, int level = 0) noexcept(false);
	void BindRBO(Renderbuffer& rbo, GLenum buffer) noexcept(false);

	bool IsComplete() const;

	void Delete();

	static void Blit(Framebuffer& src, Framebuffer& dst, GLenum mask, GLenum interp, glm::uvec2 src0, glm::uvec2 src1, glm::uvec2 dst0, glm::uvec2 dst1);
};
} // gl_engine //

#endif // OPENGL_FRAME_BUFFER_CLASS_H //