#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

uniform float size;

out vec3 vert_color;

void main() {
    vert_color = aColor;
    gl_Position = vec4(size * aPos, 1.0f);
}