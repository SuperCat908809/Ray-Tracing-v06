#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

out vec3 model_pos;
out vec3 world_pos;
out vec3 normal;
out vec2 tex_coord;


uniform float scale;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;


void main() {
	model_pos = aPos * scale;
	world_pos = vec3(model * vec4(model_pos, 1.0f));
	gl_Position = proj * view * vec4(world_pos, 1.0f);

	normal = aNormal;
	tex_coord = aTexCoord;
}