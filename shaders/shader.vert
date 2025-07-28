#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 faceNormal;
layout(location = 0) out vec3 fragColor;

layout(set = 1, binding = 0) uniform UniformBufferObject {
	mat4 mat;
} model;
layout(set = 1, binding = 1) uniform ProjMat {
	mat4 mat;
} proj;

// MARK: Lighting
const vec4 POSITIVE_Z = vec4(0.0, 0.0, 1.0, 0.0);

void main() {
	gl_Position = proj.mat * model.mat * vec4(inPosition, 1.0);
	vec4 adj_face_normal = proj.mat * model.mat * vec4(faceNormal, 0.0);

	float lighting_alignment = clamp(dot(POSITIVE_Z, adj_face_normal), 0.0, 1.0);	
	float light = 0.1 + (0.9 * lighting_alignment);
	fragColor = vec3(0.0, light, 0.0);
}
