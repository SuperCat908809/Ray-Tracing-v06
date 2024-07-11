#version 330 core
out vec4 frag_color;

uniform vec4 color;

in vec2 frag_tex;

uniform sampler2D tex0;

void main() {
    frag_color = texture(tex0, frag_tex);
}