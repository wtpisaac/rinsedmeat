package main

/*
rinsedmeat - voxel engine prototyping
copyright (c) 2025 Isaac Trimble-Pederson, All Rights Reserved
*/

import "core:log"
import "core:math"
import "core:math/linalg"
import "core:mem"
import "core:os"
import "core:os/os2"
import "core:path/filepath"
import "core:strings"
import "core:testing"
import "vendor:sdl3"

// MARK: Perspective Projection
/*
	NOTE: Perspective Projection from World Coordinates to Vulkan NDC

	WARN: SDL GPU API has its own coordinate conventions not necessarily
	aligned with the Vulkan specification. I should keep this in mind if and when
	I transition to using the Vulkan API for my work.

	SDL's coordinates are defined as follows:
	NDC -> Centered around (0, 0), bottom left (-1, -1), top right (1, 1)
	Viewport -> Top left (0, 0) bottom right (viewportWidth, viewportHeight)
	Texture coordinates -> Top-left (0,0) bottom-right (1, 1) (+Y down)
	Source: https://wiki.libsdl.org/SDL3/CategoryGPU

	Real Time Rendering describes a view frustum
	The view frustum ranges from a near plane to a far plane
	(l, b, n) to (r, b, n) describes the coordinates of the near plane in the view space
	I want to analyze this projection matrix - unsure what it is doing exactly.	

	// NOTE: General Remarks
	// I honestly did not know where to start on this. However, reading the Wikipedia article on FOV
	// helped to give me some context, leading me to this article
	// https://en.wikipedia.org/wiki/Field_of_view_in_video_games
	// This article mentions "Hor+" and this is the approach I have tried to model.
	// I did the trig by hand, have decided to try treating the near plane *as* the screen,
	// then we pass in a desired vfov to lock into, and then make the l/r differ by the screen res.
	// I then took a perspective projection matrix from DirectX's example described in Real Time Rendering
	// given SDL's coordinate system seems to model DirectX's (on my cursory inspection of the left handed
	// coordinate space, and the [0, 1] z range.
*/

make_perspective_matrix :: proc(
	near: f32,
	far: f32,
	vfov_deg: f32,
	screen_w_res: f32,
	screen_h_res: f32,
) -> matrix[4, 4]f32 {
	// Take screen resolution, near, compute r and l
	// We assume the vertical distance to be d=1, and take the horizontal as a multiplier (its aspect ratio)
	// This is independent of resolution, because we are converting into NDC, not raw pixels.
	vfov_rad := math.to_radians_f32(vfov_deg)
	horizontal_per_vertical := screen_w_res / screen_h_res
	top := near * math.tan_f32(vfov_rad / 2)
	bottom := -top
	// TODO: Will this work...
	right := top * horizontal_per_vertical
	left := -right

	// NOTE: Source matrix in Real Time Rendering, section 4.7 projections, DirectX
	return {
		(2 * near) / (right - left),
		0,
		0,
		0,
		0,
		(2 * near) / (top - bottom),
		0,
		0,
		// NOTE: We are not using the x/y alterations of the z-coordinate, because our frusta are
		// not presently asymmetric, and these will always resolve to zero.
		// If we were to e.g. support VR, Real Time Rendering indicates we might want to alter this
		// to accept asymmetric frustums.
		0,
		0,
		// NOTE: Using the DirectX z-plane calculations, given we have a [0, 1] z range,
		// not OpenGL [-1, 1], and should not need the mirroring since the axis works positive
		// for further.
		far / (far - near),
		-1 * ((far * near) / (far - near)),
		0,
		0,
		1,
		0,
	}
}

// MARK: Camera
make_camera_matrix :: proc(position: [3]f32, rotation_y_deg: f32) -> matrix[4, 4]f32 {
	translation_matrix := matrix[4, 4]f32{
		1, 0, 0, -position.x, 
		0, 1, 0, -position.y, 
		0, 0, 1, -position.z, 
		0, 0, 0, 1, 
	}
	rotation_y_rad := math.to_radians_f32(rotation_y_deg)
	rotation_inv_matrix := matrix[4, 4]f32{
		math.cos_f32(rotation_y_rad), 0, math.sin_f32(rotation_y_rad), 0, 
		0, 1, 0, 0, 
		-1 * math.sin_f32(rotation_y_rad), 0, math.cos_f32(rotation_y_rad), 0, 
		0, 0, 0, 1, 
	}
	rotation_matrix := linalg.transpose(rotation_inv_matrix)

	return rotation_matrix * translation_matrix
}

// MARK: Lighting Computations

// NOTE: We need face normals for a simplified lighting model. This will enable us to visually inspect
// a rotating cube and make sense of it, to spot check our perspective projection. We will simulate
// a directional floodlight by taking a dot product of the normal directions and the relevant
// z-direction.
// TODO: Can this be done using a compute shader?
make_normals :: proc(vertices: []f32, normalize: bool = true) -> []f32 {
	assert(
		len(vertices) % 9 == 0,
		"Incoming vertices must be % 9 == 0, must be three-dim * three per face.",
	)
	triangle_count := len(vertices) / 9

	// Allocate destination array 
	// Each face should get one normal
	normals := make([]f32, len(vertices))

	for i in 0 ..< triangle_count {
		// NOTE: Take a cross product from two vectors
		// The first vector is the first vertex to the second
		// The second vector is the second vertex to the third
		base_idx := i * 9
		vAB: [3]f32 = {
			vertices[base_idx + 3] - vertices[base_idx + 0],
			vertices[base_idx + 4] - vertices[base_idx + 1],
			vertices[base_idx + 5] - vertices[base_idx + 2],
		}
		vBC: [3]f32 = {
			vertices[base_idx + 6] - vertices[base_idx + 3],
			vertices[base_idx + 7] - vertices[base_idx + 4],
			vertices[base_idx + 8] - vertices[base_idx + 5],
		}
		cross_vec := linalg.vector_cross3(vAB, vBC)
		if (normalize) {
			cross_vec = linalg.vector_normalize0(cross_vec)
		}
		// TODO: Do we really need to send three identical data points to the GPU? Case for indexing?
		normals[base_idx] = cross_vec[0]
		normals[base_idx + 3] = cross_vec[0]
		normals[base_idx + 6] = cross_vec[0]
		normals[base_idx + 1] = cross_vec[1]
		normals[base_idx + 4] = cross_vec[1]
		normals[base_idx + 7] = cross_vec[1]
		normals[base_idx + 2] = cross_vec[2]
		normals[base_idx + 5] = cross_vec[2]
		normals[base_idx + 8] = cross_vec[2]
	}

	return normals
}

@(test)
test_normal_computations_basic :: proc(t: ^testing.T) {
	vertices: []f32 = {
		// First triangle
		0.0,
		0.0,
		0.0,
		1.0,
		0.0,
		0.0,
		0.0,
		1.0,
		0.0,
		// Second triangle
		0.0,
		0.0,
		0.0,
		2.0,
		0.0,
		0.0,
		0.0,
		0.0,
		2.0,
	}

	results := make_normals(vertices)

	testing.expect_value(t, len(results), 18)
	testing.expect_value(t, results[0], 0.0)
	testing.expect_value(t, results[1], 0.0)
	testing.expect_value(t, results[2], 1.0)
	// Further are repeats. Eventually we should refactor to compress this to an identifier.
	testing.expect_value(t, results[3], 0.0)
	testing.expect_value(t, results[4], 0.0)
	testing.expect_value(t, results[5], 1.0)
	testing.expect_value(t, results[6], 0.0)
	testing.expect_value(t, results[7], 0.0)
	testing.expect_value(t, results[8], 1.0)


	testing.expect_value(t, results[9], 0.0)
	testing.expect_value(t, results[10], -1.0)
	testing.expect_value(t, results[11], 0.0)
	testing.expect_value(t, results[12], 0.0)
	testing.expect_value(t, results[13], -1.0)
	testing.expect_value(t, results[14], 0.0)
	testing.expect_value(t, results[15], 0.0)
	testing.expect_value(t, results[16], -1.0)
	testing.expect_value(t, results[17], 0.0)

}

@(test)
test_normal_computations_normalization_off :: proc(t: ^testing.T) {
	vertices: []f32 = {0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0}

	results := make_normals(vertices, normalize = false)

	testing.expect_value(t, len(results), 9)
	testing.expect_value(t, results[0], 0.0)
	testing.expect_value(t, results[1], -4.0)
	testing.expect_value(t, results[2], 0.0)
	testing.expect_value(t, results[3], 0.0)
	testing.expect_value(t, results[4], -4.0)
	testing.expect_value(t, results[5], 0.0)
	testing.expect_value(t, results[6], 0.0)
	testing.expect_value(t, results[7], -4.0)
	testing.expect_value(t, results[8], 0.0)

}

make_normal_matrix :: proc(model: matrix[4, 4]f32, cam: matrix[4, 4]f32) -> matrix[3, 3]f32 {
	mat3 :: distinct matrix[3, 3]f32
	cam3 := mat3(cam)
	model3 := mat3(model)
	mcp := cam3 * model3
	return linalg.transpose(linalg.matrix3_adjoint(mcp))
}

// MARK: SDL Fatal Error Handling

HaltingErrorSource :: enum {
	UNKNOWN,
	CUSTOM,
	SDL,
}

HaltPrintingMessage :: proc(message: string, source: HaltingErrorSource = .UNKNOWN) -> ! {
	log.error("A FATAL ERROR HAS OCCURRED. THE PROGRAM WILL NOW HALT.")
	log.error(message)
	switch source {
	case .UNKNOWN:
		log.warn(
			"No source for this error was defined. SDL's GetError will be logged below - do not trust this error, but attempt to ascertain if the error likely arose from an SDL call and use this as context",
		)
		log.warn(sdl3.GetError())
	case .SDL:
		log.error(sdl3.GetError())
	case .CUSTOM:
		break
	}
	os.exit(1)
}

// MARK: Configuration
Configuration :: struct {
	resolution: struct {
		window_height: uint,
		window_width:  uint,
	},
}

EngineState :: struct {
	resolution:        struct {
		h: i32,
		w: i32,
	},
	test_mesh:         Maybe(ActiveMesh),
	// MARK: SDL3 GPU (Device, Shaders, etc.)
	gpu:               ^sdl3.GPUDevice,
	vertex_shader:     ^sdl3.GPUShader,
	fragment_shader:   ^sdl3.GPUShader,
	graphics_pipeline: ^sdl3.GPUGraphicsPipeline,
}

// MARK: Mesh Management

// Lots of unknowns here. Seems we are intended to submit meshes once and then do point updates on necessary metadata.
// Given this, we probably want to be able to submit meshes and load/unload them at will.
// We also need to associate a model-to-world space conversion.
// In a proper engine this is probably hierarchical in a tree - let's KISS and keep a one-level flat hierarchy for now
// will need a tree if/when we use this to render other objects (e.g., enemies)

Scene :: struct {
	gpu:           ^sdl3.GPUDevice,
	active_meshes: [dynamic]ActiveMesh,
}

SceneInit :: proc(gpu: ^sdl3.GPUDevice) -> Scene {
	return Scene{gpu = gpu, active_meshes = {}}
}

ActiveMesh :: struct {
	// TODO: Does this need a generation? or can we get away with just throwing meshes into the scene arbitrarily?
	gpu_buffer:         ^sdl3.GPUBuffer,
	normals_gpu_buffer: ^sdl3.GPUBuffer,
	model_to_world_mat: matrix[4, 4]f32,
	vertex_count:       u32,
	normals:            []f32,
}

// TODO: Make this attach to the Scene, or ditch the Scene concept for the prototype.
@(require_results)
StateRegisterMesh :: proc(
	state: ^EngineState,
	vertices: []f32,
	normals: []f32,
	model_to_world_mat: matrix[4, 4]f32,
) -> (
	mesh: ActiveMesh,
	ok: bool,
) {
	log.debugf("REGISTERING VERTS %v WITH NORMALS %v", vertices, normals)
	assert(
		len(vertices) % 9 == 0,
		"Provided vertex buffer failed modulo 9 check; must provide full triangles with 3-dim coordinates.",
	)
	assert(len(normals) == len(vertices), "Number of normals does not match len(vertices)")

	// MARK: Create the GPU buffer
	buffer_create_info := sdl3.GPUBufferCreateInfo {
		usage = sdl3.GPUBufferUsageFlags{.VERTEX},
		size  = u32(size_of(f32) * len(vertices)),
	}
	buffer := sdl3.CreateGPUBuffer(state.gpu, buffer_create_info)
	if buffer == nil {
		log.errorf(
			"Could not create GPU buffer for vertices due to SDL error. %v",
			sdl3.GetError(),
		)
		ok = false
		return
	}
	normal_buffer_create_info := sdl3.GPUBufferCreateInfo {
		usage = sdl3.GPUBufferUsageFlags{.VERTEX},
		size  = u32(size_of(f32) * len(normals)),
	}
	normal_buffer := sdl3.CreateGPUBuffer(state.gpu, normal_buffer_create_info)
	if normal_buffer == nil {
		log.errorf(
			"Could not create GPU buffer for face normals due to SDL error. %v",
			sdl3.GetError(),
		)
		ok = false
		return
	}

	// Transfer the data into the buffer
	// We do not cycle what is in this buffer, so cycling does not matter yet.
	// We should revisit this... can we reuse a fixed number of GPU buffers for chunks and utilize cycling?
	transfer_buffer_create_info := sdl3.GPUTransferBufferCreateInfo {
		usage = .UPLOAD,
		size  = u32(size_of(f32) * len(vertices)),
	}
	transfer_buffer := sdl3.CreateGPUTransferBuffer(state.gpu, transfer_buffer_create_info)
	if transfer_buffer == nil {
		ok = false
		return
	}
	defer {sdl3.ReleaseGPUTransferBuffer(state.gpu, transfer_buffer)}

	normals_transfer_buffer_create_info := sdl3.GPUTransferBufferCreateInfo {
		usage = .UPLOAD,
		size  = u32(size_of(f32) * len(normals)),
	}
	normals_transfer_buffer := sdl3.CreateGPUTransferBuffer(
		state.gpu,
		normals_transfer_buffer_create_info,
	)
	if normals_transfer_buffer == nil {
		ok = false
		return
	}
	defer {sdl3.ReleaseGPUTransferBuffer(state.gpu, normals_transfer_buffer)}

	transfer_map_loc := sdl3.MapGPUTransferBuffer(state.gpu, transfer_buffer, false)
	if transfer_map_loc == nil {
		ok = false
		return
	}
	mem.copy(transfer_map_loc, raw_data(vertices), len(vertices) * size_of(f32))

	normals_transfer_map_loc := sdl3.MapGPUTransferBuffer(
		state.gpu,
		normals_transfer_buffer,
		false,
	)
	if normals_transfer_map_loc == nil {
		ok = false
		return
	}
	mem.copy(normals_transfer_map_loc, raw_data(normals), len(normals) * size_of(f32))

	// Create a command buffer for submitting the copy
	command_buffer := sdl3.AcquireGPUCommandBuffer(state.gpu)
	if command_buffer == nil {
		ok = false
		return
	}
	copy_pass := sdl3.BeginGPUCopyPass(command_buffer)
	transfer_buffer_loc := sdl3.GPUTransferBufferLocation {
		transfer_buffer = transfer_buffer,
		offset          = 0,
	}
	gpu_buffer_region := sdl3.GPUBufferRegion {
		buffer = buffer,
		offset = 0,
		size   = u32(len(vertices) * size_of(f32)),
	}
	sdl3.UploadToGPUBuffer(copy_pass, transfer_buffer_loc, gpu_buffer_region, false)

	normals_transfer_buffer_loc := sdl3.GPUTransferBufferLocation {
		transfer_buffer = normals_transfer_buffer,
		offset          = 0,
	}
	normals_gpu_buffer_region := sdl3.GPUBufferRegion {
		buffer = normal_buffer,
		offset = 0,
		size   = u32(len(normals) * size_of(f32)),
	}
	sdl3.UploadToGPUBuffer(
		copy_pass,
		normals_transfer_buffer_loc,
		normals_gpu_buffer_region,
		false,
	)
	sdl3.EndGPUCopyPass(copy_pass)

	submit_success := sdl3.SubmitGPUCommandBuffer(command_buffer)
	if !submit_success {
		ok = false
		return
	}

	active_mesh := ActiveMesh {
		gpu_buffer         = buffer,
		normals_gpu_buffer = normal_buffer,
		model_to_world_mat = model_to_world_mat,
		vertex_count       = u32(len(vertices) / 3),
		normals            = normals,
	}
	return active_mesh, true
}

// TODO: SceneDeleteMesh

// MARK: Rendering
draw_frame :: proc(state: EngineState, window: ^sdl3.Window) {
	log.debug("Acquiring command buffer for frame")
	gpu_command_buffer := sdl3.AcquireGPUCommandBuffer(state.gpu)
	if (gpu_command_buffer == nil) {
		HaltPrintingMessage("Command buffer acquisition failed.", source = .SDL)
	}
	log.debug("Command buffer acquired.")

	// NOTE: Swapchain texture acquisition managed by SDL3 - should not free this texture
	// See: https://wiki.libsdl.org/SDL3/SDL_WaitAndAcquireGPUSwapchainTexture#remarks
	log.debug("Acquiring swapchain texture for command buffer")
	swapchain_tex: ^sdl3.GPUTexture
	swapchain_tex_width: ^u32
	swapchain_tex_height: ^u32
	swapchain_tex_success := sdl3.WaitAndAcquireGPUSwapchainTexture(
		gpu_command_buffer,
		window,
		&swapchain_tex,
		swapchain_tex_width,
		swapchain_tex_height,
	)
	if !swapchain_tex_success {
		HaltPrintingMessage("Failed to acquire GPU swapchain texture.", source = .SDL)
	}
	log.debug("Swapchain texture acquired.")

	log.debug("Executing render pass.")
	gpu_color_targets := []sdl3.GPUColorTargetInfo {
		{
			texture = swapchain_tex,
			clear_color = {0.0, 0.0, 1.0, 1.0},
			load_op = sdl3.GPULoadOp.CLEAR,
			store_op = sdl3.GPUStoreOp.STORE,
		},
	}

	// MARK: Render Pass
	gpu_render_pass := sdl3.BeginGPURenderPass(
		gpu_command_buffer,
		raw_data(gpu_color_targets),
		1,
		nil,
	)

	sdl3.BindGPUGraphicsPipeline(gpu_render_pass, state.graphics_pipeline)

	sdl3.SetGPUViewport(
		gpu_render_pass,
		sdl3.GPUViewport {
			x         = 0.0, // f32(state.resolution.w) / 2.0,
			y         = 0.0, // f32(state.resolution.h) / 2.0,
			w         = f32(state.resolution.w),
			h         = f32(state.resolution.h),
			min_depth = 0.0,
			max_depth = 1.0,
		},
	)

	// FIXME: Move off test mesh into generalizable logic
	test_mesh := state.test_mesh.?

	// TODO: What exactly is happening with the GPU Vertex Buffers?
	// It seems like it is possible to pass multiple vertex buffers into one command...
	// is that sensible? What is the advantage? Because we need different calls for uniforms...
	// maybe for instancing?
	sdl3.BindGPUVertexBuffers(
		gpu_render_pass,
		0, // TODO: Check slot
		raw_data(
			[]sdl3.GPUBufferBinding {
				sdl3.GPUBufferBinding{buffer = test_mesh.gpu_buffer, offset = 0},
				sdl3.GPUBufferBinding{buffer = test_mesh.normals_gpu_buffer, offset = 0},
			},
		),
		2,
	)
	// TODO: Should this be moved somewhere else?
	sdl3.PushGPUVertexUniformData(
		gpu_command_buffer,
		0, // TODO: Make this not a magic number!
		raw_data(&test_mesh.model_to_world_mat),
		size_of(test_mesh.model_to_world_mat), // TODO: Make this not a magic number for matrix element count
	)
	// TODO: Push perspective matrix
	perspective_matrix := make_perspective_matrix(
		1.0,
		20,
		// NOTE: vFOV currently hack from 90 deg hFOV on 16:9 via below calculator.
		// https://themetalmuncher.github.io/fov-calc/
		// TODO: Validate above, and figure out the math!
		47,
		f32(state.resolution.w),
		f32(state.resolution.h),
	)
	log.debugf("perspective matrix: %v", perspective_matrix)
	sdl3.PushGPUVertexUniformData(
		gpu_command_buffer,
		1,
		raw_data(&perspective_matrix),
		size_of(perspective_matrix),
	)
	camera_matrix := make_camera_matrix({3, 0, 0}, -30)
	log.debugf("camera matrix: %v", camera_matrix)
	sdl3.PushGPUVertexUniformData(
		gpu_command_buffer,
		2,
		raw_data(&camera_matrix),
		size_of(camera_matrix),
	)
	normal_matrix := make_normal_matrix(test_mesh.model_to_world_mat, camera_matrix)
	sdl3.PushGPUVertexUniformData(
		gpu_command_buffer,
		3,
		raw_data(&normal_matrix),
		size_of(normal_matrix),
	)
	// FIXME: Remove this debug log
	/*
	debug_verts := make([]f32, test_mesh.vertex_count, context.temp_allocator)
	for i in 0 ..< (test_mesh.vertex_count / 3) {
		vec: [3]f32 = {
			test_mesh.normals[i * 3],
			test_mesh.normals[i * 3 + 1],
			test_mesh.normals[i * 3 + 2],
		}
		transformed := normal_matrix * vec
		debug_verts[i * 3] = transformed[0]
		debug_verts[i * 3 + 1] = transformed[1]
		debug_verts[i * 3 + 2] = transformed[2]
	}
	log.debugf("transformed normals %v", debug_verts)
	*/
	sdl3.DrawGPUPrimitives(gpu_render_pass, test_mesh.vertex_count, 1, 0, 0)
	log.debug("Do we make it here?")
	log.debugf("%v", test_mesh.vertex_count)
	sdl3.EndGPURenderPass(gpu_render_pass)
	log.debug("Render pass executed.")

	log.debug("Submitting command buffer.")
	gpu_command_buffer_submit_success := sdl3.SubmitGPUCommandBuffer(gpu_command_buffer)
	if !gpu_command_buffer_submit_success {
		HaltPrintingMessage("Submission of command buffer to GPU failed.", source = .SDL)
	}
}

// MARK: Test Scene

// NOTE: Praise the cube!
register_test_mesh :: proc(state: ^EngineState) {
	TEST_MESH_VERTICES: []f32 = {
		// Front L
		-0.5,
		-0.5,
		-0.5,
		-0.5,
		0.5,
		-0.5,
		0.5,
		0.5,
		-0.5,
		// Front R
		0.5,
		0.5,
		-0.5,
		0.5,
		-0.5,
		-0.5,
		-0.5,
		-0.5,
		-0.5,
		// Right L
		0.5,
		-0.5,
		-0.5,
		0.5,
		0.5,
		-0.5,
		0.5,
		0.5,
		0.5,
		// Right R
		0.5,
		0.5,
		0.5,
		0.5,
		-0.5,
		0.5,
		0.5,
		-0.5,
		-0.5,
		// Back L
		0.5,
		-0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		-0.5,
		0.5,
		0.5,
		// Back R
		-0.5,
		0.5,
		0.5,
		-0.5,
		-0.5,
		0.5,
		0.5,
		-0.5,
		0.5,
		// Left L
		-0.5,
		-0.5,
		0.5,
		-0.5,
		0.5,
		0.5,
		-0.5,
		0.5,
		-0.5,
		// Left R
		-0.5,
		0.5,
		-0.5,
		-0.5,
		-0.5,
		-0.5,
		-0.5,
		-0.5,
		0.5,
		// Top L
		-0.5,
		0.5,
		-0.5,
		-0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		// Top R
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		-0.5,
		-0.5,
		0.5,
		-0.5,
		// Bottom L
		-0.5,
		-0.5,
		0.5,
		-0.5,
		-0.5,
		-0.5,
		0.5,
		-0.5,
		-0.5,
		// Bottom R
		0.5,
		-0.5,
		-0.5,
		0.5,
		-0.5,
		0.5,
		-0.5,
		-0.5,
		0.5,
	}
	test_mesh_normals := make_normals(TEST_MESH_VERTICES)
	log.debugf("mesh normals: %v", test_mesh_normals)

	model_to_world_matrix := matrix[4, 4]f32{
		1.0, 0.0, 0.0, 0.0, 
		0.0, 1.0, 0.0, 0.0, 
		0.0, 0.0, 1.0, 4.0, 
		0.0, 0.0, 0.0, 1.0, 
	}

	mesh, ok := StateRegisterMesh(
		state,
		TEST_MESH_VERTICES,
		test_mesh_normals,
		model_to_world_matrix,
	)
	if (!ok) {
		HaltPrintingMessage("Could not register test mesh due to SDL error", source = .SDL)
	}

	state.test_mesh = mesh
}

// MARK: Main Loop

main :: proc() {
	// MARK: Tracking Allocator boilerplate
	// https://gist.github.com/karl-zylinski/4ccf438337123e7c8994df3b03604e33
	when ODIN_DEBUG {
		track: mem.Tracking_Allocator
		mem.tracking_allocator_init(&track, context.allocator)
		context.allocator = mem.tracking_allocator(&track)

		defer {
			if len(track.allocation_map) > 0 {
				fmt.eprintf("=== %v allocations not freed: ===\n", len(track.allocation_map))
				for _, entry in track.allocation_map {
					fmt.eprintf("- %v bytes @ %v\n", entry.size, entry.location)
				}
			}
			if len(track.bad_free_array) > 0 {
				fmt.eprintf("=== %v incorrect frees: ===\n", len(track.bad_free_array))
				for entry in track.bad_free_array {
					fmt.eprintf("- %p @ %v\n", entry.memory, entry.location)
				}
			}
			mem.tracking_allocator_destroy(&track)
		}
	}

	// Static configuration
	// TODO: Load configuration from the disk or environment.
	configuration := Configuration {
		resolution = {window_width = 1280, window_height = 720},
	}
	state := EngineState {
		resolution = {
			w = i32(configuration.resolution.window_width),
			h = i32(configuration.resolution.window_height),
		},
	}
	// Logging
	context.logger = log.create_console_logger(allocator = context.temp_allocator)
	log.info("rinsedmeat - engine demo created by Isaac Trimble-Pederson")

	log.infof(
		"Resolution - %v w x %v h",
		configuration.resolution.window_width,
		configuration.resolution.window_height,
	)

	// Get program executable directory 
	proc_info, proc_info_err := os2.current_process_info(
		os2.Process_Info_Fields{.Executable_Path},
		allocator = context.temp_allocator,
	)
	if proc_info_err != nil {
		HaltPrintingMessage("Unexpected error fetching process info. Quitting.", source = .CUSTOM)
	}

	prog_path := strings.clone(proc_info.executable_path)
	os2.free_process_info(proc_info, allocator = context.temp_allocator)
	prog_dir := filepath.dir(prog_path)

	// initialize SDL window
	// thanks for losing my code!!! should've used git!!!
	sdl3.SetHint("SDL_RENDER_VULKAN_DEBUG", "1")
	init_ok := sdl3.Init(sdl3.InitFlags{.VIDEO, .EVENTS})
	if (!init_ok) {HaltPrintingMessage("SDL could not initialize with .VIDEO and .EVENTS. Are you running this in a limited (non-GUI) environment?", source = .SDL)}

	// Enable Vulkan Validation Hints
	sdl3.SetHint("SDL_RENDER_VULKAN_DEBUG", "1")
	log.debug("Vulkan Validation Layers are ACTIVE")

	// initialize SDL3 window
	main_window := sdl3.CreateWindow(
		"rinsedmeat",
		i32(state.resolution.w),
		i32(state.resolution.h),
		sdl3.WindowFlags{.RESIZABLE},
	)
	if (main_window == nil) {
		HaltPrintingMessage(
			"Main window creation failed. The game cannot run without a window.",
			source = .SDL,
		)
	}

	// MARK: GPU Setup
	/*
	NOTE: GPU support is limited to Vulkan. Other platforms will require creating relevant shaders and 
	updating the below.
	*/
	log.debug("Initializing GPU device.")
	gpu := sdl3.CreateGPUDevice(
		sdl3.GPUShaderFormat{sdl3.GPUShaderFormatFlag.SPIRV},
		true,
		"vulkan",
	)
	if gpu == nil {
		HaltPrintingMessage("GPU device initialization was not successful.", source = .SDL)
	}
	state.gpu = gpu
	log.debug("GPU device initialized.")

	log.debug("Claiming to main window...")
	gpu_window_claim_success := sdl3.ClaimWindowForGPUDevice(gpu, main_window)
	if !gpu_window_claim_success {
		HaltPrintingMessage("Main window could not claim GPU device.", source = .SDL)
	}
	log.debug("Main window claimed for GPU.")

	// MARK: Loading shaders
	log.debug("Loading shaders...")

	log.debugf("Program path %v", prog_path)
	vertex_shader_path, _ := filepath.join({prog_dir, "/shaders/shader.vert.spv"})
	log.debugf("Loading vertex shader from %v", vertex_shader_path)
	vertex_shader_contents, vertex_shader_read_ok := os.read_entire_file(vertex_shader_path)
	if !vertex_shader_read_ok {
		HaltPrintingMessage("Could not load vertex shader.", source = .CUSTOM)
	}
	vertex_shader_create_info := sdl3.GPUShaderCreateInfo {
		code_size           = len(vertex_shader_contents),
		code                = raw_data(vertex_shader_contents),
		entrypoint          = "main",
		format              = sdl3.GPUShaderFormat{.SPIRV},
		stage               = .VERTEX,
		num_uniform_buffers = 4,
	}
	vertex_shader := sdl3.CreateGPUShader(gpu, vertex_shader_create_info)
	if vertex_shader == nil {
		HaltPrintingMessage("Could not create the vertex shader", source = .SDL)
	}

	fragment_shader_path, _ := filepath.join({prog_dir, "/shaders/shader.frag.spv"})
	log.debugf("Loading fragment shader from %v", fragment_shader_path)
	fragment_shader_contents, fragment_shader_read_ok := os.read_entire_file(fragment_shader_path)
	if !fragment_shader_read_ok {
		HaltPrintingMessage("Could not load fragment shader from disk.", source = .CUSTOM)
	}
	fragment_shader_create_info := sdl3.GPUShaderCreateInfo {
		code_size  = len(fragment_shader_contents),
		code       = raw_data(fragment_shader_contents),
		entrypoint = "main",
		format     = sdl3.GPUShaderFormat{.SPIRV},
		stage      = .FRAGMENT,
	}
	fragment_shader := sdl3.CreateGPUShader(gpu, fragment_shader_create_info)
	if fragment_shader == nil {
		HaltPrintingMessage("Could not create the fragment shader", source = .SDL)
	}

	state.vertex_shader = vertex_shader
	state.fragment_shader = fragment_shader

	log.debug("Shaders created!")

	// MARK: Create GPU graphics pipeline
	log.debug("Creating graphics pipeline...")
	graphics_pipeline_create_info := sdl3.GPUGraphicsPipelineCreateInfo {
		vertex_shader = vertex_shader,
		fragment_shader = fragment_shader,
		vertex_input_state = sdl3.GPUVertexInputState {
			vertex_buffer_descriptions = raw_data(
				[]sdl3.GPUVertexBufferDescription {
					sdl3.GPUVertexBufferDescription {
						slot = 0,
						pitch = 3 * size_of(f32),
						input_rate = .VERTEX,
					},
					sdl3.GPUVertexBufferDescription {
						slot = 1,
						pitch = 3 * size_of(f32),
						input_rate = .VERTEX,
					},
				},
			),
			num_vertex_buffers = 2,
			vertex_attributes = raw_data(
				[]sdl3.GPUVertexAttribute {
					sdl3.GPUVertexAttribute {
						location = 0,
						buffer_slot = 0,
						format = .FLOAT3,
						offset = 0,
					},
					sdl3.GPUVertexAttribute {
						location = 1,
						buffer_slot = 1,
						format = .FLOAT3,
						offset = 0,
					},
				},
			),
			num_vertex_attributes = 2,
		},
		primitive_type = .TRIANGLELIST,
		rasterizer_state = sdl3.GPURasterizerState {
			fill_mode = .FILL,
			cull_mode = .BACK,
			front_face = .CLOCKWISE,
			enable_depth_clip = true,
		},
		target_info = sdl3.GPUGraphicsPipelineTargetInfo {
			color_target_descriptions = raw_data(
				[]sdl3.GPUColorTargetDescription {
					sdl3.GPUColorTargetDescription{format = sdl3.GPUTextureFormat.B8G8R8A8_UNORM},
				},
			),
			num_color_targets = 1,
		},
	}
	graphics_pipeline := sdl3.CreateGPUGraphicsPipeline(gpu, graphics_pipeline_create_info)
	if graphics_pipeline == nil {
		HaltPrintingMessage("Could not create graphics pipeline.", source = .SDL)
	}
	state.graphics_pipeline = graphics_pipeline
	log.debug("Graphics pipeline created.")

	// MARK: Test Mesh Registration
	register_test_mesh(&state)

	// MARK: Event Loop
	should_keep_running := true
	for should_keep_running {
		event: sdl3.Event
		should_process_event := sdl3.PollEvent(&event)
		if (should_process_event) {
			if event.type == .QUIT {
				should_keep_running = false
			}
			if event.type == .WINDOW_RESIZED || event.type == .WINDOW_PIXEL_SIZE_CHANGED {
				// TODO: Reset any relevant GPU state here

				// MARK: Update properties from window
				w: i32
				h: i32
				ok := sdl3.GetWindowSizeInPixels(main_window, &w, &h)
				if !ok {
					log.warn(
						"Could not obtain window size in pixels despite resolution change - not failing but artifacts may be expected!",
					)
				}
				state.resolution.h = h
				state.resolution.w = w
			}
		}

		draw_frame(state, main_window)

		// Clear allocator at end of frame
		// free_all(context.temp_allocator)
	}

	log.info("Engine shutdown complete!")
}
