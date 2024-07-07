#version 330

in vec3 frag_position;
in vec4 frag_color;
in vec3 frag_normal;

uniform vec3 light_position;
uniform vec3 light2_position;
uniform vec3 light3_position;
uniform vec3 light3_color;
uniform vec3 view_position;
uniform bool light2_enabled;
uniform bool light3_enabled;

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
    // Segunda luz
    if (light2_enabled) {
        vec3 Kd2 = vec3(1.0, 0.0, 0.0); // Coeficiente de reflexión difusa de la segunda luz (rojo)
        vec3 Ld2 = vec3(3.0, 0.5, 0.5); // Intensidad de la segunda luz (más rojo)
        
        vec3 to_light2 = light2_position - frag_position;
        vec3 light2_dir = normalize(to_light2);
        float diff2 = max(dot(normal, light2_dir), 0.0);
        vec3 diffuse2 = Kd2 * Ld2 * diff2;

        vec3 reflect_dir2 = reflect(-light2_dir, normal);
        float spec2 = pow(max(dot(view_dir, reflect_dir2), 0.0), 32);
        vec3 specular2 = spec2 * Ld2; // Componente especular de la segunda luz

        diffuse += diffuse2;
        specular += specular2;
    }

    // Tercera luz
    if (light3_enabled) {
        vec3 Kd3 = vec3(0.0, 0.0, 1.0); // Coeficiente de reflexión difusa de la tercera luz (azul)
        vec3 Ld3 = light3_color * 1.0; // Intensidad de la tercera luz (color cambiante)
        
        vec3 to_light3 = light3_position - frag_position;
        vec3 light3_dir = normalize(to_light3);
        float diff3 = max(dot(normal, light3_dir), 0.0);
        vec3 diffuse3 = Kd3 * Ld3 * diff3;

        vec3 reflect_dir3 = reflect(-light3_dir, normal);
        float spec3 = pow(max(dot(view_dir, reflect_dir3), 0.0), 32);
        vec3 specular3 = spec3 * Ld3; // Componente especular de la tercera luz

        diffuse += diffuse3;
        specular += specular3;
    }

    outColor = frag_color * vec4(ambient + diffuse + specular, 1.0);
}