#version 330 core

layout (location = 0) out vec4 frag_color;

in vec3 model_pos;
in vec3 model_normal;
in vec3 world_pos;
in vec3 world_normal;
in vec2 tex_coord;

uniform sampler2D tex0;
uniform sampler2D tex1;
uniform vec3 camera_pos;
uniform vec3 light_pos;
uniform vec4 light_color;

void main() {

	vec3 normal = normalize(world_normal);
	vec3 light_dir = normalize(light_pos - world_pos);

	float ambient = 0.20f;

	vec4 albedo = texture(tex0, tex_coord);
	float diffuse_coefficient = max(dot(normal, light_dir), 0.0f);
	vec3 diffuse = albedo.rgb * diffuse_coefficient;

	float specular_light = 0.50f;
	vec3 view_dir = normalize(camera_pos - world_pos);
	vec3 reflection_dir = reflect(-light_dir, normal);
	float specular_coefficient = pow(max(dot(view_dir, reflection_dir), 0.0f), 16);
	specular_coefficient *= specular_light;
	vec3 specular = texture(tex1, tex_coord).rrr * specular_coefficient;

	frag_color = light_color * vec4(specular + diffuse * (1 + ambient), 1.0f);
}