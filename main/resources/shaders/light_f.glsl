#version 330 core

layout (location = 0) out vec4 frag_color;

in vec3 model_pos;
in vec3 model_normal;
in vec3 world_pos;
in vec3 world_normal;
in vec2 tex_coord;

uniform vec4 light_color;


void main() {
	frag_color = light_color;
}