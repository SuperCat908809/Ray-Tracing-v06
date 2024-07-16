#include "../pch.h"
#include "gl_framebuffer.h"
#include "gl_texture.h"
#include "gl_renderbuffer.h"

using namespace gl_engine;


Framebuffer::Framebuffer() : id(0), width(0), height(0) {}
Framebuffer::~Framebuffer() {
	_delete();
}
void Framebuffer::Delete() {
	_delete();
}
void Framebuffer::_delete() {
	if (id != 0)
		glDeleteFramebuffers(1, &id);
	id = 0;
}
Framebuffer::Framebuffer(Framebuffer&& other) {
	id = other.id;
	width = other.width;
	height = other.height;

	other.id = 0;
}
Framebuffer& Framebuffer::operator=(Framebuffer&& other) {
	_delete();

	id = other.id;
	width = other.width;
	height = other.height;

	other.id = 0;

	return *this;
}

Framebuffer* Framebuffer::Make(uint32_t width, uint32_t height) {
	Framebuffer* fb = new Framebuffer();

	fb->width = width;
	fb->height = height;

	glGenFramebuffers(1, &fb->id);

	return fb;
}

void Framebuffer::Bind(GL_FB_BIND_STATE binder) {
	GLenum glenum;
	switch (binder) {
	case BOTH: GL_FRAMEBUFFER; break;
	case DRAW: GL_DRAW_BUFFER; break;
	case READ: GL_READ_BUFFER; break;
	}
	glBindFramebuffer(glenum, id);
}
void Framebuffer::ClearBinding() {
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Framebuffer::BindTexture(Texture& tex, GL_FB_BUFFER buffer, int slot, int level) noexcept(false) {
	if (tex.width != width || tex.height != width) {
		throw std::runtime_error("Texture does not match framebuffer resolution");
	}
	if ((buffer ^ COLOR) == 0 && slot != 0) {
		throw std::runtime_error("Cannot bind multiple non color attachments");
	}

	GLenum glbuffer;
	if (buffer == DEPTH) glbuffer = GL_DEPTH_ATTACHMENT;
	if (buffer == STENCIL) glbuffer = GL_STENCIL_ATTACHMENT;
	if ((buffer ^ DEPTH ^ STENCIL) == 0) glbuffer = GL_DEPTH_STENCIL_ATTACHMENT;
	if (buffer == COLOR) glbuffer = GL_COLOR_ATTACHMENT0 + slot;

	Bind(BOTH);
	glFramebufferTexture2D(GL_FRAMEBUFFER, glbuffer, GL_TEXTURE_2D, tex.id, level);
}

void Framebuffer::BindRBO(Renderbuffer& rbo, GL_FB_BUFFER buffer, int slot) noexcept(false) {
	if (rbo.width != width || rbo.height != height) {
		throw std::runtime_error("Renderbuffer does not match framebuffer resolution");
	}
	if ((buffer ^ COLOR) == 0 && slot != 0) {
		throw std::runtime_error("Cannot bind multiple non color attachments");
	}

	GLenum glbuffer;
	if (buffer == DEPTH) glbuffer = GL_DEPTH_ATTACHMENT;
	if (buffer == STENCIL) glbuffer = GL_STENCIL_ATTACHMENT;
	if ((buffer ^ DEPTH ^ STENCIL) == 0) glbuffer = GL_DEPTH_STENCIL_ATTACHMENT;
	if (buffer == COLOR) glbuffer = GL_COLOR_ATTACHMENT0 + slot;

	Bind(BOTH);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo.id);
}

bool Framebuffer::IsComplete() const { return glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE; }

void Framebuffer::Blit(Framebuffer& src, Framebuffer& dst, GL_FB_BUFFER mask, GL_FB_INTERPOLATION interp, glm::uvec2 src0, glm::uvec2 src1, glm::uvec2 dst0, glm::uvec2 dst1) {
	GLbitfield glmask;
	if ((mask & COLOR) != 0) glmask |= GL_COLOR_BUFFER_BIT;
	if ((mask & DEPTH) != 0) glmask |= GL_DEPTH_BUFFER_BIT;
	if ((mask & STENCIL) != 0) glmask |= GL_STENCIL_BUFFER_BIT;

	GLenum glinterp;
	glinterp = (interp == NEAREST) ? GL_NEAREST : GL_LINEAR;

	src.Bind(READ);
	dst.Bind(DRAW);
	glBlitFramebuffer(
		src0.x, src0.y, src1.x, src1.y,
		dst0.x, dst0.y, dst1.x, dst1.y,
		glmask, glinterp
	);
}