package main

/*
rinsedmeat - voxel engine prototyping
copyright (c) 2025 Isaac Trimble-Pederson, All Rights Reserved
*/

import "core:log"
import "core:mem"
import "core:os"
import "core:os/os2"
import "core:path/filepath"
import "core:strings"
import "vendor:sdl3"

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
	model_to_world_mat: matrix[4, 4]f32,
}

@(require_results)
SceneRegisterMesh :: proc(
	scene: ^Scene,
	vertices: []f32,
	model_to_world_mat: matrix[4, 4]f32,
) -> (
	mesh: ActiveMesh,
	ok: bool,
) {
	// MARK: Create the GPU buffer
	buffer_create_info := sdl3.GPUBufferCreateInfo {
		usage = sdl3.GPUBufferUsageFlags{.VERTEX},
		size  = u32(len(vertices)),
	}
	buffer := sdl3.CreateGPUBuffer(scene.gpu, buffer_create_info)
	if buffer == nil {
		log.errorf("Could not create GPU buffer due to SDL error. %v", sdl3.GetError())
		ok = false
		return
	}

	// Transfer the data into the buffer
	// We do not cycle what is in this buffer, so cycling does not matter yet.
	// We should revisit this... can we reuse a fixed number of GPU buffers for chunks and utilize cycling?
	transfer_buffer_create_info := sdl3.GPUTransferBufferCreateInfo {
		usage = .UPLOAD,
		size  = u32(len(vertices)),
	}
	transfer_buffer := sdl3.CreateGPUTransferBuffer(scene.gpu, transfer_buffer_create_info)
	if transfer_buffer == nil {
		ok = false
		return
	}
	defer {sdl3.ReleaseGPUTransferBuffer(scene.gpu, transfer_buffer)}

	transfer_map_loc := sdl3.MapGPUTransferBuffer(scene.gpu, transfer_buffer, false)
	if transfer_map_loc == nil {
		ok = false
		return
	}
	mem.copy(transfer_map_loc, raw_data(vertices), len(vertices) * size_of(f32))

	// Create a command buffer for submitting the copy
	command_buffer := sdl3.AcquireGPUCommandBuffer(scene.gpu)
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
		size   = size_of(f32),
	}
	sdl3.UploadToGPUBuffer(copy_pass, transfer_buffer_loc, gpu_buffer_region, false)

	submit_success := sdl3.SubmitGPUCommandBuffer(command_buffer)
	if !submit_success {
		ok = false
		return
	}

	active_mesh := ActiveMesh {
		gpu_buffer         = buffer,
		model_to_world_mat = model_to_world_mat,
	}
	return active_mesh, true
}

// TODO: SceneDeleteMesh

// MARK: Rendering
draw_frame :: proc(gpu: ^sdl3.GPUDevice, window: ^sdl3.Window) {
	log.debug("Acquiring command buffer for frame")
	gpu_command_buffer := sdl3.AcquireGPUCommandBuffer(gpu)
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
			clear_color = {0.0, 0.5, 0.5, 1.0},
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
register_test_mesh :: proc(state: ^EngineState, scene: ^Scene) {
	// TODO: Define cube meshes in model space.
	// What is our model space?
	// NOTE: Currently this is a SQUARE not a CUBE.
	TEST_MESH_VERTICES: []f32 = {
		// Front face, L
		-1.0,
		-1.0,
		1.0,
		-1.0,
		1.0,
		1.0,
		1.0,
		-1.0,
		1.0,
		// Front face, R
		1.0,
		-1.0,
		1.0,
		-1.0,
		1.0,
		1.0,
		1.0,
		1.0,
		1.0,
	}

	// TODO: Implement perspective projection matrix.
	// FIXME: Below is a scale matrix for testing. This is not permanent.
	model_to_world_matrix := matrix[4, 4]f32{
		0.5, 0.0, 0.0, 0.0, 
		0.0, 0.5, 0.0, 0.0, 
		0.0, 0.0, 0.5, 0.0, 
		0.0, 0.0, 0.0, 1.0, 
	}

	mesh, ok := SceneRegisterMesh(scene, TEST_MESH_VERTICES, model_to_world_matrix)
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
	init_ok := sdl3.Init(sdl3.InitFlags{.VIDEO, .EVENTS})
	if (!init_ok) {HaltPrintingMessage("SDL could not initialize with .VIDEO and .EVENTS. Are you running this in a limited (non-GUI) environment?", source = .SDL)}

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
		code_size  = len(vertex_shader_contents),
		code       = raw_data(vertex_shader_contents),
		entrypoint = "main",
		format     = sdl3.GPUShaderFormat{.SPIRV},
		stage      = .VERTEX,
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
						pitch = size_of(f32),
						input_rate = .VERTEX,
					},
				},
			),
			num_vertex_buffers = 1,
			vertex_attributes = raw_data(
				[]sdl3.GPUVertexAttribute {
					sdl3.GPUVertexAttribute {
						location = 0,
						buffer_slot = 0,
						format = .FLOAT2,
						offset = 0,
					},
				},
			),
			num_vertex_attributes = 1,
		},
		primitive_type = .TRIANGLELIST,
		rasterizer_state = sdl3.GPURasterizerState {
			fill_mode = .FILL,
			cull_mode = .BACK,
			front_face = .CLOCKWISE,
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

		draw_frame(gpu, main_window)
	}

	log.info("Engine shutdown complete!")
}
