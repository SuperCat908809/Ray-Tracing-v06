#version 330 core

layout (location = 0) out vec3 frag_color;

in vec2 tex_coord;

uniform sampler2D screen_tex;
uniform float split;
uniform int split_hori;

const float dx = 1.0f / 800.0f;
const float dy = 1.0f / 800.0f;

vec2 dv[9] = vec2[] (
	vec2(-dx,  dy), vec2(  0,  dy), vec2( dx,  dy),
	vec2(-dx,   0), vec2(  0,   0), vec2( dx,   0),
	vec2(-dx, -dy), vec2(  0, -dy), vec2( dx, -dy)
);

float kernel[9] = float[] (
	 1,  1,  1,
	 1, -8,  1,
	 1,  1,  1
);

void main() {

	if (split_hori == 1) {
		if (tex_coord.x < split) {
			frag_color = vec3(texture(screen_tex, tex_coord));
			return;
		}
	}
	else {
		if (tex_coord.y < split) {
			frag_color = vec3(texture(screen_tex, tex_coord));
			return;
		}
	}

	vec3 color = vec3(0.0f);

	for (int i = 0; i < 9; i++)
		color += vec3(texture(screen_tex, tex_coord + dv[i])) * kernel[i];

	frag_color = color;
}