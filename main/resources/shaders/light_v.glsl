#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

out vec3 model_pos;
out vec3 model_normal;
out vec3 world_pos;
out vec3 world_normal;
out vec2 tex_coord;

uniform mat4 model_matrix;
uniform mat4 camera_matrix;


void main() {
	model_pos = aPos;
	world_pos = vec3(model_matrix * vec4(model_pos, 1.0f));
	gl_Position = camera_matrix * vec4(world_pos, 1.0f);

	model_normal = aNormal;
	world_normal = vec3(model_matrix * vec4(model_normal, 0.0f));

	tex_coord = aTexCoord;
}