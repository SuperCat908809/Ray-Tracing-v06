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

Framebuffer* Framebuffer::DefaultFramebuffer(uint32_t width, uint32_t height) {
	Framebuffer* fb = new Framebuffer();

	fb->width = width;
	fb->height = height;
	fb->id = 0;

	return fb;
}

Framebuffer* Framebuffer::Make(uint32_t width, uint32_t height) {
	Framebuffer* fb = new Framebuffer();

	fb->width = width;
	fb->height = height;

	glGenFramebuffers(1, &fb->id);

	return fb;
}

void Framebuffer::Bind(GLenum binder) {
	glBindFramebuffer(binder, id);
}
void Framebuffer::SetDrawRegion(glm::uvec2 dimension, glm::uvec2 offset) {
	dimension += offset;
	if (dimension.x > width || dimension.y > height) {
		throw std::runtime_error("Drawing outside of framebuffer bounds");
	}
	glViewport(offset.x, offset.y, dimension.x, dimension.y);
}

void Framebuffer::BindTexture(Texture& tex, GLenum buffer, int level) noexcept(false) {
	if (tex.width != width || tex.height != width) {
		throw std::runtime_error("Texture does not match framebuffer resolution");
	}

	Bind(GL_FRAMEBUFFER);
	glFramebufferTexture2D(GL_FRAMEBUFFER, buffer, GL_TEXTURE_2D, tex.id, level);
}

void Framebuffer::BindRBO(Renderbuffer& rbo, GLenum buffer) noexcept(false) {
	if (rbo.width != width || rbo.height != height) {
		throw std::runtime_error("Renderbuffer does not match framebuffer resolution");
	}

	Bind(GL_FRAMEBUFFER);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, buffer, GL_RENDERBUFFER, rbo.id);
}

bool Framebuffer::IsComplete() const { return glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE; }

void Framebuffer::Blit(Framebuffer& src, Framebuffer& dst, GLenum buffer_mask, GLenum interp,
	glm::uvec2 src_dim, glm::uvec2 dst_dim, glm::uvec2 src_offset, glm::uvec2 dst_offset) {

	glm::uvec2 src0, src1, dst0, dst1;
	if (src_dim == glm::uvec2(0u, 0u)) {
		src1 = glm::uvec2(src.width, src.height);
	}
	if (dst_dim == glm::uvec2(0u, 0u)) {
		dst1 = glm::uvec2(dst.width, dst.height);
	}

	src0 = src_offset;
	dst0 = dst_offset;

	src1 += src0;
	dst1 += dst0;

	if (src1.x > src.width || src1.y > src.height) {
		throw std::runtime_error("Reading src framebuffer out of bounds");
	}
	if (dst1.x > dst.width || dst1.y > dst.height) {
		throw std::runtime_error("Writing dst framebuffer out of bounds");
	}

	src.Bind(GL_READ_FRAMEBUFFER);
	dst.Bind(GL_DRAW_FRAMEBUFFER);
	glBlitFramebuffer(
		src0.x, src0.y, src1.x, src1.y,
		dst0.x, dst0.y, dst1.x, dst1.y,
		buffer_mask, interp
	);
}