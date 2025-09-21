#version 450

layout(location = 0) in vec2 uv;
layout(location = 1) in float light;
layout(location = 0) out vec4 outColor;

layout(set = 2, binding = 0) uniform sampler2DArray blockTexture;

void main() {
	vec3 baseColor = texture(blockTexture, vec3(uv, 1.0)).rgb;
	vec3 modifiedColor = (0.2*light + 0.8) * baseColor;
	outColor = vec4(modifiedColor, 1.0);
}

