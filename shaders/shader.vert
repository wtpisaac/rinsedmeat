#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 0) out vec3 fragColor;

layout(set = 1, binding = 0) uniform UniformBufferObject {
	mat4 model;
} ubo;

void main() {
	gl_Position = ubo.model * vec4(inPosition, 1.0);
	fragColor = vec3(0.0, 0.0, 1.0);
}
