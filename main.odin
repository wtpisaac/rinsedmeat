package main

import "core:log"
import "core:os"
import "vendor:sdl3"

// MARK: SDL Fatal Error Handling

HaltSDLPrintingMessage :: proc(message: string) -> ! {
	log.error("A FATAL ERROR HAS OCCURRED. THE PROGRAM WILL NOW HALT.")
	log.error(message)
	log.error(sdl3.GetError())
	os.exit(1)
}

// MARK: Configuration
// WARN: Mutation of the configuration object is not supported in this prototype.
RinsedMeatConfiguration :: struct {
	resolution: struct {
		window_height: uint,
		window_width:  uint,
	},
}

// MARK: Main Loop

main :: proc() {
	// Static configuration
	// TODO: Load configuration from the disk or environment.
	configuration := RinsedMeatConfiguration {
		resolution = {window_width = 1280, window_height = 720},
	}
	// Logging
	context.logger = log.create_console_logger(allocator = context.temp_allocator)
	log.info("rinsedmeat - engine demo created by Isaac Trimble-Pederson")

	log.infof(
		"Resolution - %v w x %v h",
		configuration.resolution.window_width,
		configuration.resolution.window_height,
	)
	// initialize SDL window
	// thanks for losing my code!!! should've used git!!!
	init_ok := sdl3.Init(sdl3.InitFlags{.VIDEO, .EVENTS})
	if (!init_ok) {HaltSDLPrintingMessage("SDL could not initialize with .VIDEO and .EVENTS. Are you running this in a limited (non-GUI) environment?")}

	// initialize SDL3 window
	main_window := sdl3.CreateWindow(
		"rinsedmeat",
		i32(configuration.resolution.window_width),
		i32(configuration.resolution.window_height),
		sdl3.WindowFlags{},
	)
	if (main_window == nil) {
		HaltSDLPrintingMessage(
			"Main window creation failed. The game cannot run without a window.",
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
		HaltSDLPrintingMessage("GPU device initialization was not successful.")
	}
	log.debug("GPU device initialized.")

	log.debug("Claiming to main window...")
	gpu_window_claim_success := sdl3.ClaimWindowForGPUDevice(gpu, main_window)
	if !gpu_window_claim_success {
		HaltSDLPrintingMessage("Main window could not claim GPU device.")
	}
	log.debug("Main window claimed for GPU.")

	log.debug("Acquiring command buffer for frame")
	gpu_command_buffer := sdl3.AcquireGPUCommandBuffer(gpu)
	if (gpu_command_buffer == nil) {
		HaltSDLPrintingMessage("Command buffer acquisition failed.")
	}
	log.debug("Command buffer acquired.")

	log.debug("Acquiring swapchain texture for command buffer")
	swapchain_tex: ^sdl3.GPUTexture
	swapchain_tex_width: ^u32
	swapchain_tex_height: ^u32
	swapchain_tex_success := sdl3.WaitAndAcquireGPUSwapchainTexture(
		gpu_command_buffer,
		main_window,
		&swapchain_tex,
		swapchain_tex_width,
		swapchain_tex_height,
	)
	if !swapchain_tex_success {
		HaltSDLPrintingMessage("Failed to acquire GPU swapchain texture.")
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
		HaltSDLPrintingMessage("Submission of command buffer to GPU failed.")
	}

	should_keep_running := true
	for should_keep_running {
		event: sdl3.Event
		should_process_event := sdl3.PollEvent(&event)
		if (should_process_event) {
			if event.type == .QUIT {
				should_keep_running = false
			}
		}
	}

	log.info("Engine shutdown complete!")
}
