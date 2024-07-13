#version 330 core

layout (location = 0) out vec4 frag_color;

in vec3 model_pos;
in vec3 model_normal;
in vec3 world_pos;
in vec3 world_normal;
in vec2 tex_coord;

uniform sampler2D tex0;
uniform vec3 camera_pos;
uniform vec3 light_pos;
uniform vec4 light_color;

void main() {

	vec3 normal = normalize(world_normal);
	vec3 light_dir = normalize(light_pos - world_pos);

	float ambient = 0.20f;
	float diffuse = max(dot(normal, light_dir), 0.0f);

	float specular_light = 0.50f;
	vec3 view_dir = normalize(camera_pos - world_pos);
	vec3 reflection_dir = reflect(-light_dir, normal);
	float specular_coefficient = pow(max(dot(view_dir, reflection_dir), 0.0f), 8.0f);
	float specular = specular_light * specular_coefficient;

	vec4 albedo = texture(tex0, tex_coord);
	frag_color = albedo * light_color * (specular + diffuse + ambient);
}