#version 330 core
layout (location = 0) in vec3 aPos;
uniform float size;
void main() {
    gl_Position = vec4(size * aPos, 1.0f);
}