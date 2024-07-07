#version 330

in vec3 frag_position;
in vec4 frag_color;
in vec3 frag_normal;

uniform vec3 light_position;
uniform vec3 view_position;

out vec4 outColor;

void main()
{
    // componente difuso
    vec3 Kd = vec3(1.0, 1.0, 1.0);
    vec3 Ld = vec3(1.0, 1.0, 1.0);
    vec3 ambient = 0.1 * Ld; // componente ambiental

    vec3 normal = normalize(frag_normal);
    vec3 to_light = light_position - frag_position;
    vec3 light_dir = normalize(to_light);
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = Kd * Ld * diff;

    vec3 view_dir = normalize(view_position - frag_position);
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32);
    vec3 specular = spec * Ld; // componente especular

    outColor = frag_color * vec4(ambient + diffuse + specular, 1.0);
}



