#version 330 core
in vec3 position;
in vec3 normal;
in vec2 uv;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 transform;

out vec2 frag_texcoord;
out vec3 frag_normal;
out vec3 frag_position;

void main()
{
    frag_position = vec3(transform * vec4(position, 1.0f));
    frag_normal = mat3(transpose(inverse(transform))) * normal;
    frag_texcoord = uv;
    gl_Position = projection * view * transform * vec4(position, 1.0f);
}
