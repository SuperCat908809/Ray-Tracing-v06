#ifndef OPENGL_FRAME_BUFFER_CLASS_H
#define OPENGL_FRAME_BUFFER_CLASS_H

#include <inttypes.h>
#include <glad/glad.h>
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

	static Framebuffer* DefaultFramebuffer(uint32_t width, uint32_t height);
	static Framebuffer* Make(uint32_t width, uint32_t height);

	void Bind(GLenum binder);
	void SetDrawRegion() { SetDrawRegion({ width,height }); }
	void SetDrawRegion(glm::uvec2 dimensions, glm::uvec2 offset = glm::uvec2(0u, 0u));
	//static void ClearBinding();

	void BindTexture(Texture& tex, GLenum buffer, int level = 0) noexcept(false);
	void BindRBO(Renderbuffer& rbo, GLenum buffer) noexcept(false);

	bool IsComplete() const;

	void Delete();

	static void Blit(Framebuffer& src, Framebuffer& dst, GLenum buffer_mask, GLenum interp,
		glm::uvec2 src_dim = glm::uvec2(0u,0u),
		glm::uvec2 dst_dim = glm::uvec2(0u,0u),
		glm::uvec2 src_offset = glm::uvec2(0u,0u),
		glm::uvec2 dst_offset = glm::uvec2(0u,0u)
	);
};
} // gl_engine //

#endif // OPENGL_FRAME_BUFFER_CLASS_H //