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

import "primitives"

// MARK: Perspective Projection
/*
	NOTE: Perspective Projection from World Coordinates to Vulkan NDC

	SDL's coordinates are defined as follows:
	NDC -> Centered around (0, 0), bottom left (-1, -1), top right (1, 1)
	Viewport -> Top left (0, 0) bottom right (viewportWidth, viewportHeight)
	Texture coordinates -> Top-left (0,0) bottom-right (1, 1) (+Y down)
	Source: https://wiki.libsdl.org/SDL3/CategoryGPU

	Real Time Rendering describes a view frustum
	The view frustum ranges from a near plane to a far plane
	(l, b, n) to (r, b, n) describes the coordinates of the near plane in the view space

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
	// TODO: In rewrite, we should revisit this mathematics, and provide options if it would help for scaling
	// into different configurations.
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
CameraState :: struct {
	position: [3]f32,
	rotation: struct {
		y: f32,
		x: f32,
	},
	movement: PlayerMovement,
}

make_camera_matrix :: proc(
	position: [3]f32,
	rotation_y_deg: f32,
	rotation_x_deg: f32,
) -> matrix[4, 4]f32 {
	// NOTE: A camera matrix's purpose is to transform the world space into the camera space.
	// The "camera space" is the transformation of the space of the world into that which aligns the
	// +Z direction into that of the ray pointing outwards of the camera lens.
	// To perform this coordinate space transformation, we need to perform two actions.
	// 1. Translate the positions of the world by the opposite of that of the camera's position, thus
	// equalizing the camera and world positions by each other's difference.
	// 2. Rotate all the coordinates of this translated world by what is necessary to move their positions
	// into the desired rotation of the camera, thus bringing the world into the direction which the camera
	// is intended to rotate, with the camera remaining still, using the transpose of the matrix, which on
	// an orthogonal matrix (typical) acts as the inverse.
	translation_matrix := matrix[4, 4]f32{
		1, 0, 0, -position.x, 
		0, 1, 0, -position.y, 
		0, 0, 1, -position.z, 
		0, 0, 0, 1, 
	}
	rotation_y_rad := math.to_radians_f32(rotation_y_deg)
	rotation_y_inv_matrix := matrix[4, 4]f32{
		math.cos_f32(rotation_y_rad), 0, math.sin_f32(rotation_y_rad), 0, 
		0, 1, 0, 0, 
		-1 * math.sin_f32(rotation_y_rad), 0, math.cos_f32(rotation_y_rad), 0, 
		0, 0, 0, 1, 
	}
	rotation_x_rad := math.to_radians_f32(rotation_x_deg)
	rotation_x_inv_matrix := matrix[4, 4]f32{
		1, 0, 0, 0, 
		0, math.cos_f32(rotation_x_rad), -1 * math.sin_f32(rotation_x_rad), 0, 
		0, math.sin_f32(rotation_x_rad), math.cos_f32(rotation_x_rad), 0, 
		0, 0, 0, 1, 
	}
	// NOTE: See "Euler Angles" on Wikipedia, Rotation matrix
	// R = X(a)Y(b)Z(c)
	// https://en.wikipedia.org/wiki/Euler_angles
	// NOTE: The rotation occurred in the opposite direction I expected and rolled the camera
	// for y rotations - I believe because we are taking the transpose? In any case,
	// x then y seems to do the job correctly without the unexpected spinning.
	rotation_inv_matrix := rotation_y_inv_matrix * rotation_x_inv_matrix
	rotation_matrix := linalg.transpose(rotation_inv_matrix)

	return rotation_matrix * translation_matrix
}

camera_offset_to_world_offset :: proc(cam_y_rot_deg: f32, offset: [3]f32) -> [3]f32 {
	rotation_y_rad := math.to_radians_f32(cam_y_rot_deg)
	rotation_matrix := matrix[3, 3]f32{
		math.cos_f32(rotation_y_rad), 0, math.sin_f32(rotation_y_rad), 
		0, 1, 0, 
		-1 * math.sin_f32(rotation_y_rad), 0, math.cos_f32(rotation_y_rad), 
	}

	return rotation_matrix * offset
}

// MARK: Player Movement
// TODO: We will need the movement to be performed relative to the camera's local coordinate space,
// then after computing the local transformation, bring that into world space with the relevant matrix.
// This should not affect this particular state machine only focusing on direct movement controls.
// TODO: This naming is fucking terrible... redo it
PlayerMovement :: distinct [3]PlayerAxisMovementState

PlayerMovementAxis :: enum {
	HORIZONTAL,
	VERTICAL,
	DEPTHWISE,
}

PlayerMovementEventStatus :: enum {
	BEGAN,
	ENDED,
}

// NOTE: STILL case should not be used in movement events.
PlayerAxisMovementState :: enum {
	STILL,
	POSITIVE, // up, right, forward
	NEGATIVE, // down, left, backward
}

PlayerMovementEvent :: struct {
	axis:   PlayerMovementAxis,
	state:  PlayerAxisMovementState,
	status: PlayerMovementEventStatus,
}

process_movement_event :: proc(movement: ^PlayerMovement, event: PlayerMovementEvent) {
	assert(event.state != .STILL, "Should not submit a STILL event; this is meaningless.")

	axis_ptr: ^PlayerAxisMovementState
	switch event.axis {
	case .HORIZONTAL:
		axis_ptr = &movement.x
	case .VERTICAL:
		axis_ptr = &movement.y
	case .DEPTHWISE:
		axis_ptr = &movement.z
	}

	if event.status == .BEGAN {
		axis_ptr^ = event.state
	}
	if event.status == .ENDED && axis_ptr^ == event.state {
		axis_ptr^ = .STILL
	}
}

movement_event_for_keyboard_event :: proc(
	event: sdl3.KeyboardEvent,
) -> Maybe(PlayerMovementEvent) {
	if !(event.type == .KEY_UP || event.type == .KEY_DOWN) {
		return nil
	}
	if !(event.scancode == .W ||
		   event.scancode == .S ||
		   event.scancode == .A ||
		   event.scancode == .D ||
		   event.scancode == .LSHIFT ||
		   event.scancode == .SPACE) {
		return nil
	}

	movement_status: PlayerMovementEventStatus
	#partial switch event.type {
	case .KEY_UP:
		movement_status = .ENDED
	case .KEY_DOWN:
		movement_status = .BEGAN
	}

	movement_axis: PlayerMovementAxis
	movement_state: PlayerAxisMovementState
	#partial switch event.scancode {
	case .W:
		movement_axis = .DEPTHWISE
		movement_state = .POSITIVE
	case .S:
		movement_axis = .DEPTHWISE
		movement_state = .NEGATIVE
	case .A:
		movement_axis = .HORIZONTAL
		movement_state = .NEGATIVE
	case .D:
		movement_axis = .HORIZONTAL
		movement_state = .POSITIVE
	case .LSHIFT:
		movement_axis = .VERTICAL
		movement_state = .NEGATIVE
	case .SPACE:
		movement_axis = .VERTICAL
		movement_state = .POSITIVE
	}

	return PlayerMovementEvent {
		status = movement_status,
		axis = movement_axis,
		state = movement_state,
	}
}

num_for_state :: proc(state: PlayerAxisMovementState) -> f32 {
	n: f32
	switch state {
	case .STILL:
		n = 0
	case .NEGATIVE:
		n = -1
	case .POSITIVE:
		n = 1
	}

	return n
}

execute_movement :: proc(
	cam_pos: ^[3]f32,
	cam_rot_y: f32,
	movement: PlayerMovement,
	speed: f32,
	millis: f32,
) {
	v: [3]f32 = {num_for_state(movement.x), num_for_state(movement.y), num_for_state(movement.z)}
	v *= speed * millis
	v = camera_offset_to_world_offset(cam_rot_y, v)
	cam_pos^ += v
}

// MARK: Lighting Computations

// NOTE: We need face normals for a simplified lighting model. This will enable us to visually inspect
// a rotating cube and make sense of it, to spot check our perspective projection. We will simulate
// a directional floodlight by taking a dot product of the normal directions and the relevant
// z-direction.
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
		// The cross product thus will face away from both components of the triangle,
		// thus tangent and away from the face.
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
		// TODO: Indexing?
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

// NOTE: Some test code is below, helping to exercise the normal calculations.
// The normals are computed on the CPU, since the GPU is intended for parallel computation,
// but the computation of the normals requires understanding the vertices in chunks.
// TODO: Move the below test code to their own file.
// TODO: How would one accelerate this on the CPU? Normals could be broken into chunks and
// the computation need not be sequential - is this optimized already by the compiler?

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

// NOTE: The normal matrix is the matrix which transforms the normals of an object, which does
// not use the same rules as the positions. Namely,
// The normal vectors ought not to be translated, as normals simply reflect a direction, and 
// intermediary computations on this direction should simply rotate them.
// There are more complex models but - the inverse transpose of the model and view matrices works
// under some assumptions. We are using that for now.
make_normal_matrix :: proc(model: matrix[4, 4]f32, cam: matrix[4, 4]f32) -> matrix[4, 4]f32 {
	mat3 :: distinct matrix[3, 3]f32
	mat4 :: distinct matrix[4, 4]f32
	cam3 := mat3(cam)
	model3 := mat3(model)
	cm3 := cam3 * model3
	cm4: matrix[4, 4]f32 = mat4(cm3)
	cm4T := linalg.inverse_transpose(cm4)
	return cm4T
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
// NOTE: The configuration represents the loaded settings at the launch of the game, or to be persisted
// to disk. It should be abstracted from the present state of the engine.
Configuration :: struct {
	resolution: struct {
		window_height: uint,
		window_width:  uint,
	},
}

// MARK: Engine State
// NOTE: The engine state is the root data structure of the application, storing necessary properties. This
// would likely be broken out as the engine evolves or is rewritten. It loads initial state based on the
// configuration, stores relevant data during setup, and reads/writes to the state over the course of the
// game. 
// For this prototyping, synchronization is not implemented. If a version of the engine were to be made with
// multiple threads, breaking various subsystems sensibly would be advisable to avoid multiple threads mutating
// the same state.
EngineState :: struct {
	resolution:        struct {
		h: i32,
		w: i32,
	},
	camera:            CameraState,
	meshes:            [dynamic]ActiveMesh,
	meshes_live:       [dynamic]bool,
	// MARK: SDL3 GPU (Device, Shaders, etc.)
	gpu:               ^sdl3.GPUDevice,
	vertex_shader:     ^sdl3.GPUShader,
	fragment_shader:   ^sdl3.GPUShader,
	graphics_pipeline: ^sdl3.GPUGraphicsPipeline,
}

// MARK: Mesh Management
// Represents a mesh actively loaded into the scene. Some data is preserved for general use on the CPU, or for debugging.
// Otherwise, the main pieces are the model-to-world matrix to mutate the world space position and retain ability to
// transform the entity, the vertex counts for the draw call, and of course the reference to the GPU buffers which hold
// the mesh data(s)
ActiveMesh :: struct {
	// TODO: Will this need a generational index when we eventually evolve to submission and removal of meshes?
	live:               bool,
	gpu_buffer:         ^sdl3.GPUBuffer,
	normals_gpu_buffer: ^sdl3.GPUBuffer,
	model_to_world_mat: matrix[4, 4]f32,
	vertex_count:       u32,
}

// TODO: Make this attach to the Scene, or ditch the Scene concept for the prototype.
// It seems unlikely that we would need to load/unload distinct scenes in the prototype - and more likely that we can
// get away with a bundle of global state, so storing in the State should be relatively safe.
// In a more complete engine, we would probably have different scenes.
// NOTE: The registration of meshes has the following responsibilities, presently:
// 1. Submit the mesh vertices to the GPU. This will be stored in a GPUBuffer.
// 2. Create a transfer buffer for transferring data from the CPU to the GPU buffer.
// 3. Map the transfer buffer and copy the data.
// Since we are not updating the mesh data other than once, we do not currently make use of SDL's cycling. This may change
// under different usage scenarios.
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

	assert(
		len(state.meshes_live) == len(state.meshes),
		"Mesh bool list length out of sync with mesh data; structure corrupt",
	)

	// MARK: Create the GPU buffer
	log.debug("Creating GPU buffer for mesh vertex data...")
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
	log.debug("Vertex GPU buffer created.")

	log.debug("Creating GPU buffer for mesh normals data...")
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
	log.debug("Normals GPU buffer created.")

	// Transfer the data into the buffer
	// We do not cycle what is in this buffer, so cycling does not matter yet.
	// We should revisit this... can we reuse a fixed number of GPU buffers for chunks and utilize cycling?
	log.debug("Creating GPU transfer buffer for mesh vertex data...")
	transfer_buffer_create_info := sdl3.GPUTransferBufferCreateInfo {
		usage = .UPLOAD,
		size  = u32(size_of(f32) * len(vertices)),
	}
	transfer_buffer := sdl3.CreateGPUTransferBuffer(state.gpu, transfer_buffer_create_info)
	if transfer_buffer == nil {
		ok = false
		return
	}
	log.debug("Vertex GPU transfer buffer created.")
	defer {
		log.debug("Releasing GPU vertex transfer buffer.")
		sdl3.ReleaseGPUTransferBuffer(state.gpu, transfer_buffer)
		log.debug("GPU vertex transfer buffer released.")
	}

	log.debug("Creating GPU transfer buffer for mesh normals data...")
	normals_transfer_buffer_create_info := sdl3.GPUTransferBufferCreateInfo {
		usage = .UPLOAD,
		size  = u32(size_of(f32) * len(normals)),
	}
	normals_transfer_buffer := sdl3.CreateGPUTransferBuffer(
		state.gpu,
		normals_transfer_buffer_create_info,
	)
	if normals_transfer_buffer == nil {
		log.errorf("Could not create normals transfer buffer due to SDL error %v", sdl3.GetError())
		ok = false
		return
	}
	log.debug("Mesh normals GPU transfer buffer created.")
	defer {
		log.debug("Releasing GPU mesh normals transfer buffer...")
		sdl3.ReleaseGPUTransferBuffer(state.gpu, normals_transfer_buffer)
		log.debug("GPU mesh normals transfer buffer released.")
	}

	log.debug("Mapping vertex GPU transfer buffer to CPU.")
	transfer_map_loc := sdl3.MapGPUTransferBuffer(state.gpu, transfer_buffer, false)
	if transfer_map_loc == nil {
		log.errorf("Could not map GPU transfer buffer due to SDL error %v", sdl3.GetError())
		ok = false
		return
	}
	log.debug("Vertex GPU transfer buffer mapped. Copying...")
	mem.copy(transfer_map_loc, raw_data(vertices), len(vertices) * size_of(f32))
	log.debug("Vertex GPU transfer buffer copied.")

	log.debug("Mapping normals GPU transfer buffer to CPU.")
	normals_transfer_map_loc := sdl3.MapGPUTransferBuffer(
		state.gpu,
		normals_transfer_buffer,
		false,
	)
	if normals_transfer_map_loc == nil {
		log.errorf(
			"Could not map normals GPU transfer buffer due to SDL error %v",
			sdl3.GetError(),
		)
		ok = false
		return
	}
	log.debug("Normals GPU transfer buffer mapped. Copying...")
	mem.copy(normals_transfer_map_loc, raw_data(normals), len(normals) * size_of(f32))
	log.debug("Normals GPU transfer buffer copied.")

	// Create a command buffer for submitting the copy
	log.debug("Creating command buffer for mesh data copy")
	command_buffer := sdl3.AcquireGPUCommandBuffer(state.gpu)
	if command_buffer == nil {
		log.errorf(
			"Could not create command buffer for mesh data copy due to SDL error %v",
			sdl3.GetError(),
		)
		ok = false
		return
	}
	log.debug("Mesh data copied.")
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
		log.errorf(
			"Could not submit copy pass command buffer due to SDL error %v",
			sdl3.GetError(),
		)
		ok = false
		return
	}
	log.debug("Copy pass submitted. Returning mesh.")

	active_mesh := ActiveMesh {
		live               = true,
		gpu_buffer         = buffer,
		normals_gpu_buffer = normal_buffer,
		model_to_world_mat = model_to_world_mat,
		vertex_count       = u32(len(vertices) / 3),
	}

	// Find a slot to insert this active mesh.
	mesh_slot_count := len(state.meshes_live)
	i := 0
	for i = 0; i < mesh_slot_count; i += 1 {
		if !state.meshes_live[i] {break}
	}
	// Two possible cases:
	// 1. i exceeds current length -> append
	// 2. i is less than current length -> free slot, replace
	if i < mesh_slot_count {
		// Case 1
		state.meshes[i] = active_mesh
		state.meshes_live[i] = true
	} else {
		// Case 2
		append(&state.meshes, active_mesh)
		append(&state.meshes_live, true)
	}

	return active_mesh, true
}

// TODO: SceneDeleteMesh

// MARK: Rendering
// NOTE: Performs the frame drawing logic. Should be ran at the maximal desired framerate.
// General responsibilities include:
// 1. Obtaining the swapchain texture for the frame.
// 2. Specifying the swapchain texture as the target for drawing.
// 3. Binding to the graphics pipeline we set up at the beginning of the program.
// 4. Specifying our viewport as the window
// 5. Computing the matrix data
// 6. Rendering.
// TODO: draw_frame should be generic over the registered meshes, and should not be hardcoded
// for a single registered mesh.
draw_frame :: proc(state: EngineState, window: ^sdl3.Window) {
	gpu_command_buffer := sdl3.AcquireGPUCommandBuffer(state.gpu)
	if (gpu_command_buffer == nil) {
		HaltPrintingMessage("Command buffer acquisition failed.", source = .SDL)
	}

	// NOTE: Swapchain texture acquisition managed by SDL3 - should not free this texture
	// See: https://wiki.libsdl.org/SDL3/SDL_WaitAndAcquireGPUSwapchainTexture#remarks
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

	gpu_color_targets := []sdl3.GPUColorTargetInfo {
		{
			texture = swapchain_tex,
			clear_color = {0.1, 0.1, 0.1, 1.0},
			load_op = sdl3.GPULoadOp.CLEAR,
			store_op = sdl3.GPUStoreOp.STORE,
		},
	}
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

	for mesh in state.meshes {
		// Do not render meshes marked as dead.
		// This should be OK if we make aggressive use of slots... probably OK
		if !mesh.live {continue}

		sdl3.BindGPUVertexBuffers(
			gpu_render_pass,
			0, // TODO: Check slot
			raw_data(
				[]sdl3.GPUBufferBinding {
					sdl3.GPUBufferBinding{buffer = mesh.gpu_buffer, offset = 0},
					sdl3.GPUBufferBinding{buffer = mesh.normals_gpu_buffer, offset = 0},
				},
			),
			2,
		)
		perspective_matrix := make_perspective_matrix(
			1.0,
			20,
			// NOTE: vFOV currently hack from 90 deg hFOV on 16:9 via below calculator.
			// https://themetalmuncher.github.io/fov-calc/
			47,
			f32(state.resolution.w),
			f32(state.resolution.h),
		)
		log.debugf("perspective matrix: %v", perspective_matrix)
		camera_matrix := make_camera_matrix(
			state.camera.position,
			state.camera.rotation.y,
			state.camera.rotation.x,
		)
		log.debugf("camera matrix: %v", camera_matrix)
		mvp := perspective_matrix * camera_matrix * mesh.model_to_world_mat
		sdl3.PushGPUVertexUniformData(gpu_command_buffer, 0, raw_data(&mvp), size_of(mvp))

		normal_matrix := make_normal_matrix(mesh.model_to_world_mat, camera_matrix)
		log.debugf("normal matrix: %v", normal_matrix)
		sdl3.PushGPUVertexUniformData(
			gpu_command_buffer,
			1,
			raw_data(&normal_matrix),
			size_of(normal_matrix),
		)
		sdl3.DrawGPUPrimitives(gpu_render_pass, mesh.vertex_count, 1, 0, 0)
		log.debugf("%v", mesh.vertex_count)

	}

	sdl3.EndGPURenderPass(gpu_render_pass)

	gpu_command_buffer_submit_success := sdl3.SubmitGPUCommandBuffer(gpu_command_buffer)
	if !gpu_command_buffer_submit_success {
		HaltPrintingMessage("Submission of command buffer to GPU failed.", source = .SDL)
	}
}

// MARK: Test Scene

// NOTE: Praise the cube!
// Registers a cube mesh into the scene.
register_test_mesh :: proc(state: ^EngineState, position: [3]f32) {
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

	// Translate to (0, 0, 4) in world space.
	model_to_world_matrix := matrix[4, 4]f32{
		1.0, 0.0, 0.0, position.x, 
		0.0, 1.0, 0.0, position.y, 
		0.0, 0.0, 1.0, position.z, 
		0.0, 0.0, 0.0, 1.0, 
	}

	log.debug("Submitting test mesh using StateRegisterMesh")
	mesh, ok := StateRegisterMesh(
		state,
		TEST_MESH_VERTICES,
		test_mesh_normals,
		model_to_world_matrix,
	)
	if (!ok) {
		HaltPrintingMessage("Could not register test mesh due to SDL error", source = .SDL)
	}
	log.debug("Test mesh submitted.")
}

// MARK: Event Filter
// NOTE: Currently used solely to remove mouse motion events, to avoid queue saturation.
event_filter :: proc "c" (user_data: rawptr, event: ^sdl3.Event) -> bool {
	#partial switch event.type {
	case .MOUSE_MOTION:
		return false
	case:
		return true
	}
}

// MARK: Main Loop

main :: proc() {
	// MARK: Tracking Allocator boilerplate
	// https://gist.github.com/karl-zylinski/4ccf438337123e7c8994df3b03604e33
	// In theory this will indicate bad allocations... though we presently have some and this isn't telling me anything yet.
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
	context.logger = log.create_console_logger()
	log.info("rinsedmeat - engine demo created by Isaac Trimble-Pederson")

	log.infof(
		"Resolution - %v w x %v h",
		configuration.resolution.window_width,
		configuration.resolution.window_height,
	)

	log.debug("Obtaining executable path")
	// Get program executable directory 
	proc_info, proc_info_err := os2.current_process_info(
		os2.Process_Info_Fields{.Executable_Path},
		allocator = context.allocator,
	)
	if proc_info_err != nil {
		HaltPrintingMessage(
			"Unexpected error fetching executable path. Quitting.",
			source = .CUSTOM,
		)
	}

	prog_path := strings.clone(proc_info.executable_path)
	os2.free_process_info(proc_info, context.allocator)
	prog_dir := filepath.dir(prog_path)

	// initialize SDL window
	// thanks for losing my code!!! should've used git!!!
	sdl3.SetHint("SDL_RENDER_VULKAN_DEBUG", "1")
	init_ok := sdl3.Init(sdl3.InitFlags{.VIDEO, .EVENTS})
	if (!init_ok) {HaltPrintingMessage("SDL could not initialize with .VIDEO and .EVENTS. Are you running this in a limited (non-GUI) environment?", source = .SDL)}

	// Enable Vulkan Validation Hints
	sdl3.SetHint("SDL_RENDER_VULKAN_DEBUG", "1")

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
		num_uniform_buffers = 2,
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
					// TODO: Does this work on other platforms (e.g., macOS?)
					sdl3.GPUColorTargetDescription{format = sdl3.GPUTextureFormat.B8G8R8A8_UNORM},
				},
			),
			num_color_targets         = 1,
		},
	}
	graphics_pipeline := sdl3.CreateGPUGraphicsPipeline(gpu, graphics_pipeline_create_info)
	if graphics_pipeline == nil {
		HaltPrintingMessage("Could not create graphics pipeline.", source = .SDL)
	}
	state.graphics_pipeline = graphics_pipeline
	log.debug("Graphics pipeline created.")

	// MARK: Test Mesh Registration
	register_test_mesh(&state, {0, 0, 4})
	// WARN:: Depth testing isn't configured because we don't bind a depth stencil texture.
	// It seems Mr. Claude was correct.
	// Thus, the below line causes multiple cubes to render... need to fix... but submission works.
	// register_test_mesh(&state, {0, 0, 8})

	mouse_rel_ok := sdl3.SetWindowRelativeMouseMode(main_window, true)
	if !mouse_rel_ok {
		HaltPrintingMessage("Could not set mouse to relative mode", .SDL)
	}

	dt, dt_err := primitives.EWMADt_init(0.7)
	if dt_err != nil {
		HaltPrintingMessage(
			"Could not initialize exponentially weighted DT; smoothing out of range",
		)
	}

	// MARK: Event Loop
	sdl3.SetEventFilter(event_filter, nil)
	should_keep_running := true
	for should_keep_running {
		event: sdl3.Event
		should_process_event := sdl3.PollEvent(&event)
		if (should_process_event) {
			defer {should_process_event = sdl3.PollEvent(&event)}
			if event.type == .QUIT {
				log.debug("Quit event received")
				should_keep_running = false
			}
			if event.type == .WINDOW_RESIZED || event.type == .WINDOW_PIXEL_SIZE_CHANGED {
				log.debug("Window resize event received. Updating state.")

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
			// MARK: Camera Movement
			// TODO: This will eventually need to be a more sophisticated system intended to do
			// checks for collisions, gravity, etc.
			// For now, we simply have a noclip-capable camera that will directly map keybinds
			// to position changes per frame.
			if event.type == .KEY_DOWN || event.type == .KEY_UP {
				keyboard_event := event.key
				movement_event, is_movement_event := movement_event_for_keyboard_event(
					keyboard_event,
				).?
				if is_movement_event {
					process_movement_event(&state.camera.movement, movement_event)
					log.debugf("mov: %v", state.camera.movement)
				}
			}

		}
		// MARK: Camera mouse movement
		// HACK: We run this out of the event queue, as mouse motion saturates the queue. This is probably
		// a hack, and should be considered more carefully in a full engine.
		dx: f32
		dy: f32
		flags := sdl3.GetRelativeMouseState(&dx, &dy)

		log.debugf("dx %v", dx)
		log.debugf("dy %v", dy)

		// NOTE: Swap x, y
		// Horizontal movement should be an *x-axis* rotation
		// Vertical movement should be a *y-axis* rotation
		// Screen x, y inverted from the desired rotation axis
		normalized_dx := (dy * 10) / (f32(state.resolution.w))
		normalized_dy := (dx * 10) / (f32(state.resolution.h))

		// NOTE: Camera should be able to move up and down [-90deg, 90deg]; clamp
		// Camera should be able to spin around endlessly, wrap
		// We use these to prevent these values from becoming very large over time,
		// causing precision loss
		state.camera.rotation.x = clamp(state.camera.rotation.x + normalized_dx, -90, 90)
		state.camera.rotation.y = math.wrap(state.camera.rotation.y + normalized_dy, 360)
		log.debugf("camera rot: x %v y %v", state.camera.rotation.x, state.camera.rotation.y)

		primitives.EWMADt_record_tick(&dt)
		dt_f := primitives.EWMADt_retrieve_millis(&dt)

		execute_movement(
			&state.camera.position,
			state.camera.rotation.y,
			state.camera.movement,
			0.5,
			dt_f,
		)

		draw_frame(state, main_window)

		// Clear allocator at end of frame
		free_all(context.temp_allocator)
	}

	log.info("Engine shutdown complete!")
}
