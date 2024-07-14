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

struct PhongMaterial {
	vec3 albedo;
	vec3 specular;
	vec3 normal;
};
struct PhongLight {
	vec3 color;
	float intensity;
	float ambient;
};

vec3 phong_lighting(
	PhongMaterial mat,
	PhongLight light,
	vec3 light_dir,
	vec3 cam_dir
) {
	float diffuse_coefficient = max(dot(mat.normal, light_dir), 0.0f);
	vec3 diff = mat.albedo * diffuse_coefficient;

	float specular_light = 0.50f;
	vec3 reflect_dir = reflect(-light_dir, mat.normal);
	float specular_coefficient = pow(max(dot(cam_dir, reflect_dir), 0.0f), 16);
	vec3 spec = mat.specular * specular_light * specular_coefficient;

	return light.color * light.intensity * (spec + diff * (1 + light.ambient));
}


void main() {

	vec3 normal = normalize(world_normal);
	vec3 light_dir = normalize(light_pos - world_pos);
	vec3 cam_dir = normalize(camera_pos - world_pos);

	float ambient = 0.20f;
	vec4 albedo = texture(tex0, tex_coord);
	float specular = texture(tex1, tex_coord).r;

	float ldist = length(light_pos - world_pos);
	float a = 0.5f;
	float b = 0.05f;
	float light_intensity = 1.0f / (a * ldist * ldist + b * ldist + 1.0f);

	PhongMaterial mat;
	mat.albedo = albedo.rgb;
	mat.specular = vec3(specular);
	mat.normal = normal;

	PhongLight light;
	light.color = vec3(light_color);
	light.intensity = light_intensity;
	light.ambient = ambient;

	vec3 lighting = phong_lighting(mat, light, light_dir, cam_dir);

	frag_color = vec4(lighting, 1.0f);
}