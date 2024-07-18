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
Framebuffer::Framebuffer(Framebuffer&& other) noexcept {
	id = other.id;
	width = other.width;
	height = other.height;

	other.id = 0;
}
Framebuffer& Framebuffer::operator=(Framebuffer&& other) noexcept {
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

void Framebuffer::Bind(GLenum binder) {
	//GLenum glenum = GL_FRAMEBUFFER;
	//switch (binder) {
	//case BOTH: glenum = GL_FRAMEBUFFER; break;
	//case DRAW: glenum = GL_DRAW_BUFFER; break;
	//case READ: glenum = GL_READ_BUFFER; break;
	//}
	glBindFramebuffer(binder, id);
}
void Framebuffer::ClearBinding() {
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Framebuffer::BindTexture(Texture& tex, GLenum buffer, int level) noexcept(false) {
	if (tex.width != width || tex.height != width) {
		throw std::runtime_error("Texture does not match framebuffer resolution");
	}
	//if ((buffer == GL_DEPTH_ATTACHMENT || buffer == GL_STENCIL_ATTACHMENT || buffer == GL_DEPTH_STENCIL_ATTACHMENT) && slot != 0) {
	//	throw std::runtime_error("Cannot bind multiple non color attachments");
	//}

	//GLenum glbuffer = GL_COLOR_ATTACHMENT0;
	//if (buffer & DEPTH) glbuffer = GL_DEPTH_ATTACHMENT;
	//if (buffer & STENCIL) glbuffer = GL_STENCIL_ATTACHMENT;
	//if ((buffer ^ DEPTH ^ STENCIL) & 0) glbuffer = GL_DEPTH_STENCIL_ATTACHMENT;
	//if (buffer & COLOR) glbuffer = GL_COLOR_ATTACHMENT0 + slot;

	Bind(GL_FRAMEBUFFER);
	glFramebufferTexture2D(GL_FRAMEBUFFER, buffer, GL_TEXTURE_2D, tex.id, level);
}

void Framebuffer::BindRBO(Renderbuffer& rbo, GLenum buffer) noexcept(false) {
	if (rbo.width != width || rbo.height != height) {
		throw std::runtime_error("Renderbuffer does not match framebuffer resolution");
	}
	//if ((buffer == GL_DEPTH_ATTACHMENT || buffer == GL_STENCIL_ATTACHMENT || buffer == GL_DEPTH_STENCIL_ATTACHMENT) && slot != 0) {
	//	throw std::runtime_error("Cannot bind multiple non color attachments");
	//}

	//GLenum glbuffer = GL_COLOR_ATTACHMENT0;
	//if (buffer & DEPTH) glbuffer = GL_DEPTH_ATTACHMENT;
	//if (buffer & STENCIL) glbuffer = GL_STENCIL_ATTACHMENT;
	//if ((buffer ^ DEPTH ^ STENCIL) & 0) glbuffer = GL_DEPTH_STENCIL_ATTACHMENT;
	//if (buffer & COLOR) glbuffer = GL_COLOR_ATTACHMENT0 + slot;

	Bind(GL_FRAMEBUFFER);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, buffer, GL_RENDERBUFFER, rbo.id);
}

bool Framebuffer::IsComplete() const { return glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE; }

void Framebuffer::Blit(Framebuffer& src, Framebuffer& dst, GLenum mask, GLenum interp, glm::uvec2 src0, glm::uvec2 src1, glm::uvec2 dst0, glm::uvec2 dst1) {
	//GLbitfield glmask = 0;
	//if ((mask & COLOR) != 0) glmask |= GL_COLOR_BUFFER_BIT;
	//if ((mask & DEPTH) != 0) glmask |= GL_DEPTH_BUFFER_BIT;
	//if ((mask & STENCIL) != 0) glmask |= GL_STENCIL_BUFFER_BIT;

	//GLenum glinterp;
	//glinterp = (interp == NEAREST) ? GL_NEAREST : GL_LINEAR;

	src.Bind(GL_READ_BUFFER);
	dst.Bind(GL_DRAW_BUFFER);
	glBlitFramebuffer(
		src0.x, src0.y, src1.x, src1.y,
		dst0.x, dst0.y, dst1.x, dst1.y,
		mask, interp
	);
}