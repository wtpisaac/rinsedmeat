#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 0) out vec3 fragColor;

layout(set = 1, binding = 0) uniform UniformBufferObject {
	mat4 mat;
} model;
layout(set = 1, binding = 1) uniform ProjMat {
	mat4 mat;
} proj;

void main() {
	gl_Position = proj.mat * model.mat * vec4(inPosition, 1.0);
	fragColor = vec3(0.0, 0.0, 1.0);
}
