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
import "vendor:sdl3/image"

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

CameraMatrices :: struct {
	camera_rotation: matrix[3, 3]f32,
	world_to_camera: matrix[4, 4]f32,
}

make_camera_matrices :: proc(
	position: [3]f32,
	rotation_y_deg: f32,
	rotation_x_deg: f32,
) -> CameraMatrices {
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
	camera_rotation_matrix := linalg.transpose((matrix[3, 3]f32)(rotation_inv_matrix))
	rotation_matrix := linalg.transpose(rotation_inv_matrix)

	return CameraMatrices {
		world_to_camera = rotation_matrix * translation_matrix,
		camera_rotation = camera_rotation_matrix,
	}
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

// MARK: Camera Ray Box Intersection Implementation
Ray :: struct {
	origin:    [3]f32,
	direction: [3]f32,
	// Precomputed values for optimization
	invdir:    [3]f32,
	sign:      [3]int,
}

Ray_init :: proc(origin: [3]f32, direction: [3]f32) -> Ray {
	invdir := 1 / direction

	return Ray {
		origin = origin,
		direction = direction,
		invdir = invdir,
		sign = {invdir.x < 0 ? 1 : 0, invdir.y < 0 ? 1 : 0, invdir.z < 0 ? 1 : 0},
	}
}

AABoundingBox :: struct {
	min_corner: [3]f32,
	max_corner: [3]f32,
}

RayBoxIntersectResult :: struct {
	intersects:   bool,
	intersection: [3]f32,
	intersect_t:  f32,
}

ray_box_intersect :: proc(ray: Ray, box: AABoundingBox) -> RayBoxIntersectResult {
	bounds: [2][3]f32 = {box.min_corner, box.max_corner}

	t_min := (bounds[ray.sign.x].x - ray.origin.x) * ray.invdir.x
	t_max := (bounds[1 - ray.sign.x].x - ray.origin.x) * ray.invdir.x
	ty_min := (bounds[ray.sign.y].y - ray.origin.y) * ray.invdir.y
	ty_max := (bounds[1 - ray.sign.y].y - ray.origin.y) * ray.invdir.y

	if (t_min > ty_max) || (ty_min > t_max) {return {intersects = false}}

	if ty_min > t_min {t_min = ty_min}
	if ty_max < t_max {t_max = ty_max}

	tz_min := (bounds[ray.sign.z].z - ray.origin.z) * ray.invdir.z
	tz_max := (bounds[1 - ray.sign.z].z - ray.origin.z) * ray.invdir.z

	if (t_min > tz_max) || (tz_min > t_max) {return {intersects = false}}

	if tz_min > t_min {t_min = tz_min}
	if tz_max < t_max {t_max = tz_max}

	return {
		intersects = true,
		intersection = ray.origin + ray.direction * t_min,
		intersect_t = t_min,
	}
}


@(test)
test_ray_box_intersect_intersects :: proc(t: ^testing.T) {
	box := AABoundingBox {
		min_corner = {-0.5, -0.5, 0.5},
		max_corner = {0.5, 0.5, 1.0},
	}
	ray := Ray_init(origin = {0, 0, 0}, direction = {0, 0, 4})

	intersect_results := ray_box_intersect(ray, box)

	testing.expect(t, intersect_results.intersects, "Did not intersect when expected.")
}

@(test)
test_ray_box_intersect_does_not_intersect :: proc(t: ^testing.T) {
	box := AABoundingBox {
		min_corner = {-0.5, -0.5, 0.5},
		max_corner = {0.5, 0.5, 1.0},
	}
	ray := Ray_init(origin = {0, 0, 0}, direction = {4, 0, 0})

	intersect_results := ray_box_intersect(ray, box)

	testing.expect(t, !intersect_results.intersects, "Intersected when expected to miss.")
}

// MARK: Player Movement
// TODO: This naming is fucking terrible... redo it
PlayerMovement :: distinct [3]PlayerAxisMovement

// NOTE: STILL case should not be used in movement events.
PlayerAxisMovement :: enum {
	STILL,
	POSITIVE, // up, right, forward
	NEGATIVE, // down, left, backward
}

map_keyboard_to_player_movement :: proc(
	incoming_codes: [^]bool,
	incoming_mod_state: sdl3.Keymod,
) -> PlayerMovement {
	is_w := incoming_codes[sdl3.Scancode.W]
	is_s := incoming_codes[sdl3.Scancode.S]
	is_a := incoming_codes[sdl3.Scancode.A]
	is_d := incoming_codes[sdl3.Scancode.D]
	is_spc := incoming_codes[sdl3.Scancode.SPACE]
	is_shift: bool = .LSHIFT in incoming_mod_state

	new_movement: PlayerMovement = {}
	if is_w {new_movement.z = .POSITIVE}
	if is_s {new_movement.z = .NEGATIVE}
	if is_a {new_movement.x = .NEGATIVE}
	if is_d {new_movement.x = .POSITIVE}
	if is_spc {new_movement.y = .POSITIVE}
	if is_shift {new_movement.y = .NEGATIVE}

	return new_movement
}

num_for_state :: proc(state: PlayerAxisMovement) -> f32 {
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
	// log.debugf("len(vertices) == %v", len(vertices))
	assert(
		len(vertices) % 15 == 0,
		"Incoming vertices must be % 15 == 0, must be three-dim * three per face (+ 2-dim, 2 tex coords per face, ignored).",
	)
	// NOTE: triangle_count = vertices array / (triangle pitch per vertex * 3 vertices * 3 verts per triangle)
	triangle_count := len(vertices) / (5 * 3)

	// Allocate destination array 
	// Each face should get one normal
	normals := make([]f32, triangle_count * 9)
	// log.debugf("len(normals) == %v", len(normals))

	for i in 0 ..< triangle_count {
		// NOTE: Take a cross product from two vectors
		// The first vector is the first vertex to the second
		// The second vector is the second vertex to the third
		// The cross product thus will face away from both components of the triangle,
		// thus tangent and away from the face.
		base_idx := i * 15
		normals_base_idx := i * 9
		vAB: [3]f32 = {
			vertices[base_idx + 5] - vertices[base_idx + 0],
			vertices[base_idx + 6] - vertices[base_idx + 1],
			vertices[base_idx + 7] - vertices[base_idx + 2],
		}
		vBC: [3]f32 = {
			vertices[base_idx + 10] - vertices[base_idx + 5],
			vertices[base_idx + 11] - vertices[base_idx + 6],
			vertices[base_idx + 12] - vertices[base_idx + 7],
		}
		cross_vec := linalg.vector_cross3(vAB, vBC)
		if (normalize) {
			cross_vec = linalg.vector_normalize0(cross_vec)
		}
		// TODO: Indexing?
		normals[normals_base_idx] = cross_vec[0]
		normals[normals_base_idx + 3] = cross_vec[0]
		normals[normals_base_idx + 6] = cross_vec[0]
		normals[normals_base_idx + 1] = cross_vec[1]
		normals[normals_base_idx + 4] = cross_vec[1]
		normals[normals_base_idx + 7] = cross_vec[1]
		normals[normals_base_idx + 2] = cross_vec[2]
		normals[normals_base_idx + 5] = cross_vec[2]
		normals[normals_base_idx + 8] = cross_vec[2]
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
		// DUMMY TEXTURE COORDS
		0.0,
		0.0,
		1.0,
		0.0,
		0.0,
		// DUMMY TEXTURE COORDS
		0.0,
		0.0,
		0.0,
		1.0,
		0.0,
		// DUMMY TEXTURE COORDS
		0.0,
		0.0,
		// Second triangle
		0.0,
		0.0,
		0.0,
		// DUMMY TEXTURE COORDS
		0.0,
		0.0,
		2.0,
		0.0,
		0.0,
		// DUMMY TEXTURE COORDS
		0.0,
		0.0,
		0.0,
		0.0,
		2.0,
		// DUMMY_TEXTURE_COORDS
		0.0,
		0.0,
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
	resolution:                     struct {
		h: i32,
		w: i32,
	},
	camera:                         CameraState,
	meshes:                         [dynamic]ActiveMesh,
	meshes_live:                    [dynamic]bool,
	// MARK: SDL3 GPU (Device, Shaders, etc.)
	gpu:                            ^sdl3.GPUDevice,
	vertex_shader:                  ^sdl3.GPUShader,
	fragment_shader:                ^sdl3.GPUShader,
	graphics_pipeline:              ^sdl3.GPUGraphicsPipeline,
	sdl_keystate:                   [^]bool,
	preferred_depth_texture_format: sdl3.GPUTextureFormat,
	depth_texture:                  ^sdl3.GPUTexture,
	block_texture:                  ^sdl3.GPUTexture,
	block_texture_sampler:          ^sdl3.GPUSampler,
	block_data:                     ^BlockData,
	chunk_mesh_id:                  map[[2]int]int,
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
	slot:               int,
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
	// NOTE: "vertices" includes two texture coordinates after the verts
	// (vertX, vertY, vertZ, texU, texV)
	vertices: []f32,
	normals: []f32,
	model_to_world_mat: matrix[4, 4]f32,
) -> (
	mesh: ActiveMesh,
	ok: bool,
) {
	log.debugf("REGISTERING VERTS %v WITH NORMALS %v", vertices, normals)
	log.debugf("len(vertices) == %v, len(normals) == %v", len(vertices), len(normals))
	assert(
		len(vertices) % 15 == 0,
		"Provided vertex buffer failed modulo 15 check; must provide full triangles with 3-dim coordinates, 2-dim texture coordinates.",
	)
	assert(
		len(normals) % 9 == 0,
		"Normals failed modulo 9 check; must provide appropriate number of normals",
	)
	assert(
		len(normals) / 3 == len(vertices) / 5,
		"len(normals) == %v does not correspond with len(vertices) == %v",
	)

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
		vertex_count       = u32(len(vertices) / 5),
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
	active_mesh.slot = i

	return active_mesh, true
}

StateDeleteMesh :: proc(state: ^EngineState, slot: int) -> bool {
	state.meshes[slot].live = false
	state.meshes_live[slot] = false

	sdl3.ReleaseGPUBuffer(state.gpu, state.meshes[slot].gpu_buffer)
	sdl3.ReleaseGPUBuffer(state.gpu, state.meshes[slot].normals_gpu_buffer)

	return true
}

// MARK: Texture Management, Loading
// NOTE: Texture details on Linux on my Framework, under Hyprland/Wayland:
// [DEBUG] --- [2025-08-30 02:37:49] [main.odin:694:load_game_textures_to_gpu()] Texture format: ABGR8888
// [DEBUG] --- [2025-08-30 02:37:49] [main.odin:1081:get_preferred_depth_texture()] Depth textures will use D32_FLOAT
load_block_textures_to_gpu :: proc(device: ^sdl3.GPUDevice, paths: []string) -> ^sdl3.GPUTexture {
	surfaces := make([]^sdl3.Surface, len(paths))
	for i in 0 ..< len(paths) {
		surfaces[i] = image.Load(strings.clone_to_cstring(paths[i], context.temp_allocator))
		if surfaces[i] == nil {
			HaltPrintingMessage("Error loading game texture.", .SDL)
		}

		// Assert all textures have equivalent sizes, pitches, and formats.
		// This is true for n = 0, but must be asserted for n > 0
		if (i > 0) {
			assert(
				surfaces[i].w == surfaces[i - 1].w && surfaces[i].h == surfaces[i - 1].h,
				"textures dissimilar dimensions",
			)
			assert(surfaces[i].pitch == surfaces[i - 1].pitch, "textures dissimilar pitch")
			assert(surfaces[i].format == surfaces[i - 1].format, "textures dissimilar format")
		}
	}

	log.debugf("Texture format: %v", surfaces[0].format)
	log.debugf("Layer count: %v", len(surfaces))

	// Load into texture array
	// TODO: How do we copy into each distinct layer?
	// TODO: Does the copy screw up the colors? (probably?)

	gpu_tex_create_info := sdl3.GPUTextureCreateInfo {
		type                 = .D2_ARRAY,
		format               = .R8G8B8A8_UNORM,
		usage                = {.SAMPLER},
		width                = u32(surfaces[0].w),
		height               = u32(surfaces[0].h),
		layer_count_or_depth = u32(len(surfaces)),
		num_levels           = 1,
	}
	gpu_tex := sdl3.CreateGPUTexture(device, gpu_tex_create_info)
	if gpu_tex == nil {
		HaltPrintingMessage("Could not allocate GPU texture for block textures", .SDL)
	}

	// Copy contents into the texture
	tex_transfer_buffer_create_info := sdl3.GPUTransferBufferCreateInfo {
		// bytes per tex * texture count
		size  = 4 * u32(surfaces[0].w) * u32(surfaces[0].h) * u32(len(paths)),
		usage = .UPLOAD,
	}
	tex_transfer_buffer := sdl3.CreateGPUTransferBuffer(device, tex_transfer_buffer_create_info)
	tex_transfer_ptr := sdl3.MapGPUTransferBuffer(device, tex_transfer_buffer, false)
	if tex_transfer_ptr == nil {
		HaltPrintingMessage("Could not map GPU transfer buffer for block textures", .SDL)
	}
	// Copy textures
	for i in 0 ..< len(surfaces) {
		// We need to copy each row individually due to the pitch potentially offsetting
		for j in 0 ..< surfaces[i].h {
			// Assuming there is no pitch to the GPU texture rows. Thus we need to map
			// from some offset for the pitch, to one that simply accounts for pixels
			// per row in a contiguous block.
			// NOTE: Divide pitch by bytes - offset is per 4 bytes
			row_offset := (surfaces[i].pitch / 4) * i32(j)
			// NOTE: Do not mulitply by bytes for similar reasons, offset accounts for it.
			direct_offset := (surfaces[i].h * surfaces[i].w * i32(i)) + (surfaces[i].w * i32(j))
			mem.copy(
				mem.ptr_offset((^[4]u8)(tex_transfer_ptr), direct_offset),
				mem.ptr_offset((^[4]u8)(surfaces[i].pixels), row_offset),
				int(surfaces[i].w * 4),
			)
		}
	}

	// Unmap transfer buffer required before upload
	sdl3.UnmapGPUTransferBuffer(device, tex_transfer_buffer)

	// Copy to GPU now
	command_buffer := sdl3.AcquireGPUCommandBuffer(device)
	if command_buffer == nil {
		HaltPrintingMessage("Could not acquire command buffer for block texture transfer", .SDL)
	}

	upload_copy_pass := sdl3.BeginGPUCopyPass(command_buffer)
	for i in 0 ..< len(surfaces) {
		sdl3.UploadToGPUTexture(
			upload_copy_pass,
			sdl3.GPUTextureTransferInfo {
				transfer_buffer = tex_transfer_buffer,
				offset = u32(surfaces[i].w) * u32(surfaces[i].h) * u32(i) * 4,
			},
			sdl3.GPUTextureRegion {
				texture = gpu_tex,
				mip_level = 0,
				layer = u32(i),
				w = u32(surfaces[i].w),
				h = u32(surfaces[i].h),
				d = 1,
			},
			false,
		)
	}
	sdl3.EndGPUCopyPass(upload_copy_pass)
	ok := sdl3.SubmitGPUCommandBuffer(command_buffer)
	if !ok {
		HaltPrintingMessage("Problem encountered submitting GPU command buffer.", .SDL)
	}

	// Free relevant resources
	sdl3.ReleaseGPUTransferBuffer(device, tex_transfer_buffer)

	return gpu_tex
}

// MARK: Texture Sampling
make_block_texture_sampler :: proc(device: ^sdl3.GPUDevice) -> ^sdl3.GPUSampler {
	sampler := sdl3.CreateGPUSampler(
		device,
		sdl3.GPUSamplerCreateInfo {
			address_mode_u = .CLAMP_TO_EDGE,
			address_mode_v = .CLAMP_TO_EDGE,
			address_mode_w = .CLAMP_TO_EDGE,
		},
	)
	if sampler == nil {
		HaltPrintingMessage("Could not create block texture sampler", .SDL)
	}
	return sampler
}

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
			clear_color = {3.0 / 255.0, 194.0 / 255.0, 252.0 / 255.0, 1.0},
			load_op = sdl3.GPULoadOp.CLEAR,
			store_op = sdl3.GPUStoreOp.STORE,
		},
	}
	depth_target_info := sdl3.GPUDepthStencilTargetInfo {
		texture     = state.depth_texture,
		clear_depth = 1.0,
		load_op     = .CLEAR,
		store_op    = .DONT_CARE,
	}
	gpu_render_pass := sdl3.BeginGPURenderPass(
		gpu_command_buffer,
		raw_data(gpu_color_targets),
		1,
		&depth_target_info,
	)

	sdl3.BindGPUGraphicsPipeline(gpu_render_pass, state.graphics_pipeline)
	sdl3.BindGPUFragmentSamplers(
		gpu_render_pass,
		0,
		raw_data(
			[]sdl3.GPUTextureSamplerBinding {
				{texture = state.block_texture, sampler = state.block_texture_sampler},
			},
		),
		1,
	)

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
			100.0,
			// NOTE: vFOV currently hack from 90 deg hFOV on 16:9 via below calculator.
			// https://themetalmuncher.github.io/fov-calc/
			47,
			f32(state.resolution.w),
			f32(state.resolution.h),
		)
		camera_matrix :=
			make_camera_matrices(state.camera.position, state.camera.rotation.y, state.camera.rotation.x).world_to_camera
		mvp := perspective_matrix * camera_matrix * mesh.model_to_world_mat
		sdl3.PushGPUVertexUniformData(gpu_command_buffer, 0, raw_data(&mvp), size_of(mvp))

		normal_matrix := make_normal_matrix(mesh.model_to_world_mat, camera_matrix)
		sdl3.PushGPUVertexUniformData(
			gpu_command_buffer,
			1,
			raw_data(&normal_matrix),
			size_of(normal_matrix),
		)
		sdl3.DrawGPUPrimitives(gpu_render_pass, mesh.vertex_count, 1, 0, 0)

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
		0.0,
		1.0, // Front face, bottom-left
		-0.5,
		0.5,
		-0.5,
		0.0,
		0.0, // Front face, top-left
		0.5,
		0.5,
		-0.5,
		1.0,
		0.0, // Front face, top-right

		// Front R
		0.5,
		0.5,
		-0.5,
		1.0,
		0.0, // Front face, top-right
		0.5,
		-0.5,
		-0.5,
		1.0,
		1.0, // Front face, bottom-right
		-0.5,
		-0.5,
		-0.5,
		0.0,
		1.0, // Front face, bottom-left

		// Right L
		0.5,
		-0.5,
		-0.5,
		0.0,
		1.0, // Right face, bottom-right
		0.5,
		0.5,
		-0.5,
		0.0,
		0.0, // Right face, top-right
		0.5,
		0.5,
		0.5,
		1.0,
		0.0, // Right face, top-left

		// Right R
		0.5,
		0.5,
		0.5,
		1.0,
		0.0, // Right face, top-left
		0.5,
		-0.5,
		0.5,
		1.0,
		1.0, // Right face, bottom-left
		0.5,
		-0.5,
		-0.5,
		0.0,
		1.0, // Right face, bottom-right

		// Back L
		0.5,
		-0.5,
		0.5,
		0.0,
		1.0, // Back face, bottom-left
		0.5,
		0.5,
		0.5,
		0.0,
		0.0, // Back face, top-left
		-0.5,
		0.5,
		0.5,
		1.0,
		0.0, // Back face, top-right

		// Back R
		-0.5,
		0.5,
		0.5,
		1.0,
		0.0, // Back face, top-right
		-0.5,
		-0.5,
		0.5,
		1.0,
		1.0, // Back face, bottom-right
		0.5,
		-0.5,
		0.5,
		0.0,
		1.0, // Back face, bottom-left

		// Left L
		-0.5,
		-0.5,
		0.5,
		0.0,
		1.0, // Left face, bottom-right
		-0.5,
		0.5,
		0.5,
		0.0,
		0.0, // Left face, top-right
		-0.5,
		0.5,
		-0.5,
		1.0,
		0.0, // Left face, top-left

		// Left R
		-0.5,
		0.5,
		-0.5,
		1.0,
		0.0, // Left face, top-left
		-0.5,
		-0.5,
		-0.5,
		1.0,
		1.0, // Left face, bottom-left
		-0.5,
		-0.5,
		0.5,
		0.0,
		1.0, // Left face, bottom-right

		// Top L (texture V flipped)
		-0.5,
		0.5,
		-0.5,
		0.0,
		1.0, // Top face, top-left (V flipped)
		-0.5,
		0.5,
		0.5,
		0.0,
		0.0, // Top face, bottom-left (V flipped)
		0.5,
		0.5,
		0.5,
		1.0,
		0.0, // Top face, bottom-right (V flipped)

		// Top R (texture V flipped)
		0.5,
		0.5,
		0.5,
		1.0,
		0.0, // Top face, bottom-right (V flipped)
		0.5,
		0.5,
		-0.5,
		1.0,
		1.0, // Top face, top-right (V flipped)
		-0.5,
		0.5,
		-0.5,
		0.0,
		1.0, // Top face, top-left (V flipped)

		// Bottom L (texture V flipped)
		-0.5,
		-0.5,
		0.5,
		0.0,
		1.0, // Bottom face, top-left (V flipped)
		-0.5,
		-0.5,
		-0.5,
		0.0,
		0.0, // Bottom face, bottom-left (V flipped)
		0.5,
		-0.5,
		-0.5,
		1.0,
		0.0, // Bottom face, bottom-right (V flipped)

		// Bottom R (texture V flipped)
		0.5,
		-0.5,
		-0.5,
		1.0,
		0.0, // Bottom face, bottom-right (V flipped)
		0.5,
		-0.5,
		0.5,
		1.0,
		1.0, // Bottom face, top-right (V flipped)
		-0.5,
		-0.5,
		0.5,
		0.0,
		1.0, // Bottom face, top-left (V flipped)
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

// MARK: We have Minecraft at home
// NOTE: BLOCK DATA ZC XC Y Z X

CHUNKS_PER_HORIZONTAL :: 16
CHUNK_SIZE :: 16
CHUNK_HEIGHT :: 128
GROUND_HEIGHT :: 0
WORLD_LIMIT_HORIZONTAL :: CHUNKS_PER_HORIZONTAL * CHUNK_SIZE
WORLD_LIMIT_VERTICAL :: CHUNK_HEIGHT

BlockData :: struct {
	chunks:
	[CHUNKS_PER_HORIZONTAL][CHUNKS_PER_HORIZONTAL][CHUNK_HEIGHT][CHUNK_SIZE][CHUNK_SIZE]u16,
}

BlockData_make_and_register :: proc(state: ^EngineState) {
	BlockData_init(&state.block_data)
	BlockData_register_initial(state, state.block_data)
}

BlockData_init :: proc(data_addr: ^^BlockData) {
	// Initialize BlockData onto heap
	data := new(BlockData)

	for zc in 0 ..< CHUNKS_PER_HORIZONTAL {
		for xc in 0 ..< CHUNKS_PER_HORIZONTAL {
			for z in 0 ..< CHUNK_SIZE {
				for x in 0 ..< CHUNK_SIZE {
					data.chunks[zc][xc][0][z][x] = 1
				}
			}
		}
	}

	data_addr^ = data
}

BlockData_register_initial :: proc(state: ^EngineState, data: ^BlockData) {
	for zc in 0 ..< CHUNKS_PER_HORIZONTAL {
		for xc in 0 ..< CHUNKS_PER_HORIZONTAL {
			gpu_data := minecraft_at_home(state.block_data.chunks[zc][xc])
			model_to_world_matrix := matrix[4, 4]f32{
				1.0, 0.0, 0.0, f32(xc * CHUNK_SIZE), 
				0.0, 1.0, 0.0, GROUND_HEIGHT, 
				0.0, 0.0, 1.0, f32(zc * CHUNK_SIZE), 
				0.0, 0.0, 0.0, 1.0, 
			}
			mesh, ok := StateRegisterMesh(
				state,
				gpu_data.vertex_buffer[:],
				gpu_data.normals_buffer[:],
				model_to_world_matrix,
			)
			if !ok {
				HaltPrintingMessage("Could not register test blockdata-derived mesh.")
			}
			state.chunk_mesh_id[{zc, xc}] = mesh.slot
		}
	}
}

BlockData_reloadChunk :: proc(state: ^EngineState, zc: int, xc: int) {
	StateDeleteMesh(state, state.chunk_mesh_id[{zc, xc}])

	gpu_data := minecraft_at_home(state.block_data.chunks[zc][xc])
	model_to_world_matrix := matrix[4, 4]f32{
		1.0, 0.0, 0.0, f32(xc * CHUNK_SIZE), 
		0.0, 1.0, 0.0, GROUND_HEIGHT, 
		0.0, 0.0, 1.0, f32(zc * CHUNK_SIZE), 
		0.0, 0.0, 0.0, 1.0, 
	}
	mesh, ok := StateRegisterMesh(
		state,
		gpu_data.vertex_buffer[:],
		gpu_data.normals_buffer[:],
		model_to_world_matrix,
	)
	if !ok {
		HaltPrintingMessage("Could not register test blockdata-derived mesh.")
	}
	state.chunk_mesh_id[{zc, xc}] = mesh.slot

}

// NOTE: Block layout to do this, for now:
// [y][z][x]u16 (where u16 represents a block ID; avoiding u8 to get an idea of how
// slow this will be with room to breathe for block choices

MinecraftAtHomeResults :: struct {
	vertex_buffer:  [dynamic]f32,
	normals_buffer: [dynamic]f32,
}

// WARN: Minecraft at home:
minecraft_at_home :: proc(
	blocks: [CHUNK_HEIGHT][CHUNK_SIZE][CHUNK_SIZE]u16,
) -> MinecraftAtHomeResults {
	vertex_out_buffer: [dynamic]f32
	normals_out_buffer: [dynamic]f32

	y_chunk_limit := len(blocks)
	z_chunk_limit := len(blocks[0])
	x_chunk_limit := len(blocks[0][0])

	for y in 0 ..< len(blocks) {
		for z in 0 ..< len(blocks[y]) {
			assert(len(blocks[y]) == z_chunk_limit, "nonaligned z layer")
			for x in 0 ..< len(blocks[y][z]) {
				assert(len(blocks[y][z]) == x_chunk_limit, "nonaligned x layer")
				// If the visited block is 1, check its neighbors.
				// If the neighbors are 0, add triangles with the relevant normal vector
				// TODO: Append a texture ID to the vertex buffer
				xf := f32(x)
				yf := f32(y)
				zf := f32(z)

				if blocks[y][z][x] == 1 {
					// Check down -y (or if y at 0)
					if y == 0 || blocks[y - 1][z][x] == 0 {
						vertex_additions: []f32 = {
							// Upper UL
							xf - 0.5,
							yf - 0.5,
							zf - 0.5,
							1.0,
							0.0,
							xf + 0.5,
							yf - 0.5,
							zf - 0.5,
							0.0,
							0.0,
							xf + 0.5,
							yf - 0.5,
							zf + 0.5,
							0.0,
							1.0,

							// Lower LR
							xf + 0.5,
							yf - 0.5,
							zf + 0.5,
							0.0,
							1.0,
							xf - 0.5,
							yf - 0.5,
							zf + 0.5,
							1.0,
							1.0,
							xf - 0.5,
							yf - 0.5,
							zf - 0.5,
							1.0,
							0.0,
						}
						normals_additions := make_normals(vertex_additions)
						append(&vertex_out_buffer, ..vertex_additions)
						append(&normals_out_buffer, ..normals_additions)
					}
					// Check up +y (or if y at limit)
					if y == y_chunk_limit - 1 || blocks[y + 1][z][x] == 0 {
						vertex_additions: []f32 = {
							// Upper UL
							xf - 0.5,
							yf + 0.5,
							zf - 0.5,
							0.0,
							1.0,
							xf - 0.5,
							yf + 0.5,
							zf + 0.5,
							0.0,
							0.0,
							xf + 0.5,
							yf + 0.5,
							zf + 0.5,
							1.0,
							0.0,

							// Lower LR
							xf + 0.5,
							yf + 0.5,
							zf + 0.5,
							1.0,
							0.0,
							xf + 0.5,
							yf + 0.5,
							zf - 0.5,
							1.0,
							1.0,
							xf - 0.5,
							yf + 0.5,
							zf - 0.5,
							0.0,
							1.0,
						}
						normals_additions := make_normals(vertex_additions)
						append(&vertex_out_buffer, ..vertex_additions)
						append(&normals_out_buffer, ..normals_additions)

					}
					// Check left -x (or if x at 0
					if x == 0 || blocks[y][z][x - 1] == 0 {
						vertex_additions: []f32 = {
							// Left UL Triangle
							xf - 0.5,
							yf - 0.5,
							zf + 0.5,
							0.0,
							1.0,
							xf - 0.5,
							yf + 0.5,
							zf + 0.5,
							0.0,
							0.0,
							xf - 0.5,
							yf + 0.5,
							zf - 0.5,
							1.0,
							0.0,

							// Left LR Triangle
							xf - 0.5,
							yf + 0.5,
							zf - 0.5,
							1.0,
							0.0,
							xf - 0.5,
							yf - 0.5,
							zf - 0.5,
							1.0,
							1.0,
							xf - 0.5,
							yf - 0.5,
							zf + 0.5,
							0.0,
							1.0,
						}
						normals_additions := make_normals(vertex_additions)
						append(&vertex_out_buffer, ..vertex_additions)
						append(&normals_out_buffer, ..normals_additions)

					}
					// Check right +x (or if x at limit)
					if x == x_chunk_limit - 1 || blocks[y][z][x + 1] == 0 {
						vertex_additions: []f32 = {
							// Right UL
							xf + 0.5,
							yf - 0.5,
							zf - 0.5,
							0.0,
							1.0,
							xf + 0.5,
							yf + 0.5,
							zf - 0.5,
							0.0,
							0.0,
							xf + 0.5,
							yf + 0.5,
							zf + 0.5,
							1.0,
							0.0,

							// Lower LR
							xf + 0.5,
							yf + 0.5,
							zf + 0.5,
							1.0,
							0.0,
							xf + 0.5,
							yf - 0.5,
							zf + 0.5,
							1.0,
							1.0,
							xf + 0.5,
							yf - 0.5,
							zf - 0.5,
							0.0,
							1.0,
						}
						normals_additions := make_normals(vertex_additions)
						append(&vertex_out_buffer, ..vertex_additions)
						append(&normals_out_buffer, ..normals_additions)

					}
					// Check forward -z or if z at zero)
					if z == 0 || blocks[y][z - 1][x] == 0 {
						vertex_additions: []f32 = {
							// Front UL triangle
							xf - 0.5,
							yf - 0.5,
							zf - 0.5,
							0.0,
							1.0,
							xf - 0.5,
							yf + 0.5,
							zf - 0.5,
							0.0,
							0.0,
							xf + 0.5,
							yf + 0.5,
							zf - 0.5,
							1.0,
							0.0,

							// Front LR Triangle
							xf + 0.5,
							yf + 0.5,
							zf - 0.5,
							1.0,
							0.0,
							xf + 0.5,
							yf - 0.5,
							zf - 0.5,
							1.0,
							1.0,
							xf - 0.5,
							yf - 0.5,
							zf - 0.5,
							0.0,
							1.0,
						}
						normals_additions := make_normals(vertex_additions)
						append(&vertex_out_buffer, ..vertex_additions)
						append(&normals_out_buffer, ..normals_additions)
					}
					// Check back +z (or if z at limit)
					if z == z_chunk_limit - 1 || blocks[y][z + 1][x] == 0 {
						vertex_additions: []f32 = {
							// Back UL
							xf + 0.5,
							yf - 0.5,
							zf + 0.5,
							0.0,
							1.0,
							xf + 0.5,
							yf + 0.5,
							zf + 0.5,
							0.0,
							0.0,
							xf - 0.5,
							yf + 0.5,
							zf + 0.5,
							1.0,
							0.0,

							// Back LR
							xf - 0.5,
							yf + 0.5,
							zf + 0.5,
							1.0,
							0.0,
							xf - 0.5,
							yf - 0.5,
							zf + 0.5,
							1.0,
							1.0,
							xf + 0.5,
							yf - 0.5,
							zf + 0.5,
							0.0,
							1.0,
						}
						normals_additions := make_normals(vertex_additions)
						append(&vertex_out_buffer, ..vertex_additions)
						append(&normals_out_buffer, ..normals_additions)
					}
				}
			}
		}
	}

	return {vertex_buffer = vertex_out_buffer, normals_buffer = normals_out_buffer}
}

// MARK: Block Placement / Removal
BLOCK_PLACEMENT_RANGE :: 10

LookingAtResults :: struct {
	looking_at: Maybe([3]int),
	// NOTE: Adjacent is the block position of the block adjacent to the
	// intersected face of the camera ray.
	// WARN: Adjacent should only be populated if there is a valid,
	// in-range adjacent coordinate.
	adjacent:   Maybe([3]int),
}

camera_coordinate_to_block_coordinate :: proc(camera_position: [3]f32) -> [3]int {
	return {
		int(math.round_f32(camera_position.x)),
		int(math.round_f32(camera_position.y)),
		int(math.round_f32(camera_position.z)),
	}
}

addr_for_block :: proc(state: ^EngineState, block_position: [3]int) -> Maybe(^u16) {
	if block_position.x < 0 ||
	   block_position.x > WORLD_LIMIT_HORIZONTAL - 1 ||
	   block_position.z < 0 ||
	   block_position.z > WORLD_LIMIT_HORIZONTAL - 1 ||
	   block_position.y < 0 ||
	   block_position.y > WORLD_LIMIT_VERTICAL - 1 {
		return nil
	}
	return(
		&(state.block_data.chunks[block_position.z / CHUNK_SIZE][block_position.x / CHUNK_SIZE][block_position.y][block_position.z % CHUNK_SIZE][block_position.x % CHUNK_SIZE]) \
	)
}


compute_looking_at_block :: proc(state: ^EngineState) -> LookingAtResults {
	// We want to start a walk from the camera position, rounded to the nearest block,
	// then walk outwards. This should create a cube around the camera, probably not
	// as clean as a sphere but easiest for a first pass.

	block_coord_at_camera := camera_coordinate_to_block_coordinate(state.camera.position)
	ray_direction_vector: [3]f32 = {0.0, 0.0, 1.0}
	log.debugf("orig ray: %v", ray_direction_vector)
	ray_direction_vector *=
		make_camera_matrices(state.camera.position, state.camera.rotation.y, state.camera.rotation.x).camera_rotation
	ray_direction_vector = BLOCK_PLACEMENT_RANGE * linalg.vector_normalize(ray_direction_vector)
	log.debugf("ray: %v", ray_direction_vector)

	camera_ray := Ray_init(state.camera.position, ray_direction_vector)
	log.debugf("ray origin: %v", camera_ray.origin)

	// Iterate over the neighborhood with some radius, run the ray intersect test,
	// and pick the result (if any) that is lowest

	low_x_bound := clamp(
		block_coord_at_camera.x - BLOCK_PLACEMENT_RANGE,
		0.0,
		WORLD_LIMIT_HORIZONTAL - 1,
	)
	hi_x_bound := clamp(
		block_coord_at_camera.x + BLOCK_PLACEMENT_RANGE,
		0.0,
		WORLD_LIMIT_HORIZONTAL - 1,
	)
	low_y_bound := clamp(
		block_coord_at_camera.y - BLOCK_PLACEMENT_RANGE,
		0.0,
		WORLD_LIMIT_VERTICAL - 1,
	)
	hi_y_bound := clamp(
		block_coord_at_camera.y + BLOCK_PLACEMENT_RANGE,
		0.0,
		WORLD_LIMIT_VERTICAL - 1,
	)
	low_z_bound := clamp(
		block_coord_at_camera.z - BLOCK_PLACEMENT_RANGE,
		0.0,
		WORLD_LIMIT_HORIZONTAL - 1,
	)
	hi_z_bound := clamp(
		block_coord_at_camera.z + BLOCK_PLACEMENT_RANGE,
		0.0,
		WORLD_LIMIT_HORIZONTAL - 1,
	)

	closest_intersection: Maybe(RayBoxIntersectResult) = nil
	looking_at: [3]int

	for x in low_x_bound ..= hi_x_bound {
		for y in low_y_bound ..= hi_y_bound {
			for z in low_z_bound ..= hi_z_bound {
				// NOTE: Exclude camera itself
				if x == block_coord_at_camera.x &&
				   y == block_coord_at_camera.y &&
				   z == block_coord_at_camera.z {
					continue
				}
				if addr_for_block(state, {x, y, z}).?^ == 0 {
					continue
				}

				block_box := AABoundingBox {
					min_corner = {f32(x) - 0.5, f32(y) - 0.5, f32(z) - 0.5},
					max_corner = {f32(x) + 0.5, f32(y) + 0.5, f32(z) + 0.5},
				}
				result := ray_box_intersect(camera_ray, block_box)
				if !result.intersects {continue}
				if closest_intersection == nil {
					closest_intersection = result
					looking_at = {x, y, z}
					continue
				}
				if closest_intersection.?.intersect_t > result.intersect_t {
					closest_intersection = result
					looking_at = {x, y, z}
					continue
				}
			}
		}
	}

	if closest_intersection == nil {
		return {looking_at = nil, adjacent = nil}
	}
	intersection := closest_intersection.?

	adjacent: Maybe([3]int)
	log.debugf("T: %v", intersection.intersect_t)
	log.debugf(
		"Computed intersection coordinates: %v",
		camera_ray.origin + intersection.intersect_t * camera_ray.direction,
	)
	// Compute adjacent block by walking backwards by a small amount until the block changes
	for t := intersection.intersect_t; t > 0.0; t -= math.F32_EPSILON {
		world_coords := camera_ray.origin + t * camera_ray.direction
		block_coords := camera_coordinate_to_block_coordinate(world_coords)
		if block_coords == looking_at {continue}
		if block_coords == block_coord_at_camera {break}
		// Found adjacent?
		adjacent = block_coords
		break
	}

	return {looking_at = looking_at, adjacent = adjacent}
}

handle_place_block_action :: proc(state: ^EngineState) {
	looking_at := compute_looking_at_block(state)
	log.debugf("place, looking at %v", looking_at)
	adjacent, has_adjacent := looking_at.adjacent.?
	if !has_adjacent {return}
	adj_addr, adj_valid := addr_for_block(state, adjacent).?
	if !adj_valid {return}
	// Double check that the adjacent is 0 - I suspect bugs could be possible with this logic as-is...
	// So this is just an added check. Don't need it for the looking_at since that is inherent in
	// the result.
	if adj_addr^ == 0 {
		adj_addr^ = 1
		BlockData_reloadChunk(state, adjacent.z / CHUNK_SIZE, adjacent.x / CHUNK_SIZE)
	}
}

handle_destroy_block_action :: proc(state: ^EngineState) {
	looking_at_results := compute_looking_at_block(state)
	log.debug("destroy, looking at %v", looking_at_results)

	looking_at, has_looking_at := looking_at_results.looking_at.?
	if !has_looking_at {return}
	looking_at_addr, looking_at_valid := addr_for_block(state, looking_at).?
	if !looking_at_valid {return}
	looking_at_addr^ = 0
	BlockData_reloadChunk(state, looking_at.z / CHUNK_SIZE, looking_at.x / CHUNK_SIZE)
}

// MARK: Event Filter
// NOTE: Currently used solely to remove mouse motion events, to avoid queue saturation.
event_filter :: proc "c" (user_data: rawptr, event: ^sdl3.Event) -> bool {
	#partial switch event.type {
	case .MOUSE_MOTION, .KEY_DOWN, .KEY_UP:
		return false
	case:
		return true
	}
}

// MARK: Graphics Pipeline
create_graphics_pipeline :: proc(state: ^EngineState) {
	// MARK: Create GPU graphics pipeline
	log.debug("Creating graphics pipeline...")
	graphics_pipeline_create_info := sdl3.GPUGraphicsPipelineCreateInfo {
		vertex_shader = state.vertex_shader,
		fragment_shader = state.fragment_shader,
		vertex_input_state = sdl3.GPUVertexInputState {
			vertex_buffer_descriptions = raw_data(
				[]sdl3.GPUVertexBufferDescription {
					sdl3.GPUVertexBufferDescription {
						slot = 0,
						pitch = 5 * size_of(f32),
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
						buffer_slot = 0,
						format = .FLOAT2,
						offset = 3 * size_of(f32),
					},
					sdl3.GPUVertexAttribute {
						location = 2,
						buffer_slot = 1,
						format = .FLOAT3,
						offset = 0,
					},
				},
			),
			num_vertex_attributes = 3,
		},
		primitive_type = .TRIANGLELIST,
		rasterizer_state = sdl3.GPURasterizerState {
			fill_mode = .FILL,
			cull_mode = .BACK,
			front_face = .CLOCKWISE,
			enable_depth_clip = true,
		},
		depth_stencil_state = sdl3.GPUDepthStencilState {
			compare_op = .LESS,
			enable_depth_test = true,
			enable_depth_write = true,
		},
		target_info = sdl3.GPUGraphicsPipelineTargetInfo {
			color_target_descriptions = raw_data(
				[]sdl3.GPUColorTargetDescription {
					// TODO: Does this work on other platforms (e.g., macOS?)
					sdl3.GPUColorTargetDescription{format = sdl3.GPUTextureFormat.B8G8R8A8_UNORM},
				},
			),
			num_color_targets         = 1,
			depth_stencil_format      = state.preferred_depth_texture_format,
			has_depth_stencil_target  = true,
		},
	}
	graphics_pipeline := sdl3.CreateGPUGraphicsPipeline(state.gpu, graphics_pipeline_create_info)
	if graphics_pipeline == nil {
		HaltPrintingMessage("Could not create graphics pipeline.", source = .SDL)
	}
	state.graphics_pipeline = graphics_pipeline
	log.debug("Graphics pipeline created.")
}

// MARK: Graphics Pipeline > Depth
create_depth_texture :: proc(
	device: ^sdl3.GPUDevice,
	format: sdl3.GPUTextureFormat,
	width: u32,
	height: u32,
) -> ^sdl3.GPUTexture {
	// MARK: Depth texture configuration
	depth_texture := sdl3.CreateGPUTexture(
		device,
		sdl3.GPUTextureCreateInfo {
			type = .D2,
			format = format,
			usage = {.DEPTH_STENCIL_TARGET},
			width = width,
			height = height,
			layer_count_or_depth = 1,
			num_levels = 1,
		},
	)
	if depth_texture == nil {
		HaltPrintingMessage("Could not allocate depth texture", .SDL)
	}

	return depth_texture
}

get_preferred_depth_texture :: proc(device: ^sdl3.GPUDevice) -> sdl3.GPUTextureFormat {
	if sdl3.GPUTextureSupportsFormat(device, .D32_FLOAT, .D2, {.DEPTH_STENCIL_TARGET}) {
		log.debug("Depth textures will use D32_FLOAT")
		return .D32_FLOAT
	}

	if sdl3.GPUTextureSupportsFormat(device, .D24_UNORM, .D2, {.DEPTH_STENCIL_TARGET}) {
		log.debug("Depth textures will use D24_UNORM")
		return .D24_UNORM
	}

	HaltPrintingMessage(
		"Neither D32_FLOAT nor D24_UNORM supported. D16 is not acceptable. Quitting.",
		.CUSTOM,
	)
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
		sdl_keystate = sdl3.GetKeyboardState(nil),
		camera = {position = {0, 2, 0}},
	}
	// Logging
	context.logger = log.create_console_logger()
	context.logger.lowest_level = .Debug
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
		code_size    = len(fragment_shader_contents),
		code         = raw_data(fragment_shader_contents),
		entrypoint   = "main",
		format       = sdl3.GPUShaderFormat{.SPIRV},
		stage        = .FRAGMENT,
		num_samplers = 1,
	}
	fragment_shader := sdl3.CreateGPUShader(gpu, fragment_shader_create_info)
	if fragment_shader == nil {
		HaltPrintingMessage("Could not create the fragment shader", source = .SDL)
	}

	state.vertex_shader = vertex_shader
	state.fragment_shader = fragment_shader

	log.debug("Shaders created!")

	// MARK: Load Textures
	DEBUG_TEX_PATH :: "textures/debug.png"
	GRASS_TEX_PATH :: "textures/grass.png"
	STONE_TEX_PATH :: "textures/stone.png"

	state.block_texture = load_block_textures_to_gpu(
	state.gpu,
	{
		// NOTE: "DEBUG" texture should be at slot 0.
		// If we have block IDs, 0 -> air block, which has no texture
		// so this gives us a simple debug texture + offsets the slot for us :)
		DEBUG_TEX_PATH,
		STONE_TEX_PATH,
		GRASS_TEX_PATH,
	},
	)

	// Initialize sampler for block textures
	state.block_texture_sampler = make_block_texture_sampler(gpu)

	// MARK: Initial depth texture creation 
	// In a better engine this would be structured but I am prototyping!
	// TODO: This is a moronic thing to do inline
	state.preferred_depth_texture_format = get_preferred_depth_texture(state.gpu)
	state.depth_texture = create_depth_texture(
		state.gpu,
		state.preferred_depth_texture_format,
		u32(state.resolution.w), // TODO: these conversions are ugly as sin!
		u32(state.resolution.h),
	)
	create_graphics_pipeline(&state)

	BlockData_make_and_register(&state)

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

				// MARK: Update depth texture
				sdl3.ReleaseGPUTexture(state.gpu, state.depth_texture)
				state.depth_texture = create_depth_texture(
					state.gpu,
					state.preferred_depth_texture_format,
					u32(state.resolution.w),
					u32(state.resolution.h),
				)
			}
			if event.type == .MOUSE_BUTTON_DOWN {
				log.debugf("Mouse button %v down.", event.button.button)
				if event.button.button == 1 {
					handle_destroy_block_action(&state)
				} // left button 
				if event.button.button == 3 {
					handle_place_block_action(&state)
				} // right button
			}
		}
		// MARK: ESC to quit
		if state.sdl_keystate[sdl3.Scancode.ESCAPE] {
			should_keep_running = false
		}
		// MARK: Camera Keyboard Movement
		keyboard_modstate := sdl3.GetModState()
		state.camera.movement = map_keyboard_to_player_movement(
			state.sdl_keystate,
			keyboard_modstate,
		)

		// MARK: Camera mouse movement
		// HACK: We run this out of the event queue, as mouse motion saturates the queue. This is probably
		// a hack, and should be considered more carefully in a full engine.
		dx: f32
		dy: f32
		flags := sdl3.GetRelativeMouseState(&dx, &dy)

		// NOTE: Swap x, y
		// Horizontal movement should be an *x-axis* rotation
		// Vertical movement should be a *y-axis* rotation
		// Screen x, y inverted from the desired rotation axis
		normalized_dx := (dy * 50) / (f32(state.resolution.w))
		normalized_dy := (dx * 50) / (f32(state.resolution.h))

		// NOTE: Camera should be able to move up and down [-90deg, 90deg]; clamp
		// Camera should be able to spin around endlessly, wrap
		// We use these to prevent these values from becoming very large over time,
		// causing precision loss
		state.camera.rotation.x = clamp(state.camera.rotation.x + normalized_dx, -90, 90)
		state.camera.rotation.y = math.wrap(state.camera.rotation.y + normalized_dy, 360)

		primitives.EWMADt_record_tick(&dt)
		dt_f := primitives.EWMADt_retrieve_millis(&dt)

		execute_movement(
			&state.camera.position,
			state.camera.rotation.y,
			state.camera.movement,
			1.0,
			dt_f,
		)

		draw_frame(state, main_window)

		// Clear allocator at end of frame
		free_all(context.temp_allocator)
	}

	log.info("Engine shutdown complete!")
}
