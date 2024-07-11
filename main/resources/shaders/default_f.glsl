#version 330 core

layout (location = 0) out vec4 frag_color;

in vec3 model_pos;
in vec3 world_pos;
in vec3 normal;
in vec2 tex_coord;


uniform sampler2D tex0;

void main() {
	frag_color = texture(tex0, tex_coord);
}