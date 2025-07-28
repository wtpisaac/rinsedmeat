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
layout(set = 1, binding = 2) uniform CamMat {
	mat4 mat;
} cam;
layout(set = 1, binding = 3) uniform NormalMat {
	mat3 mat;
} norm;

// MARK: Lighting
// NOTE: We use NEGATIVE_Z to simplify alignment and avoid an add op - dot product aligning with negative
// aligns with our test lighting from camera alignment.
const vec3 NEGATIVE_Z = vec3(0.0, 0.0, -1.0);

void main() {
	gl_Position = proj.mat * cam.mat * model.mat * vec4(inPosition, 1.0);
	
	vec3 adj_face_normal = normalize(norm.mat * faceNormal);
	float lighting_alignment = clamp(dot(NEGATIVE_Z, adj_face_normal), 0.0, 1.0);
	float light = 0.1 + (0.9 * lighting_alignment);
	fragColor = vec3(0.0, light, 0.0);
	// fragColor = adj_face_normal;
}
