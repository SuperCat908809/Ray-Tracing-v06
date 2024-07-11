#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTex;

uniform float size;

out vec2 frag_tex;

void main() {
    frag_tex = aTex;
    gl_Position = vec4(size * aPos, 1.0f);
}