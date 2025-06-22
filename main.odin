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

// MARK: Main Loop

main :: proc() {
	// Logging
	context.logger = log.create_console_logger(allocator = context.temp_allocator)
	log.info("rinsedmeat - engine demo created by Isaac Trimble-Pederson")

	// initialize SDL window
	// thanks for losing my code!!! should've used git!!!
	init_ok := sdl3.Init(sdl3.InitFlags{.VIDEO, .EVENTS})
	if (!init_ok) {HaltSDLPrintingMessage("SDL could not initialize with .VIDEO and .EVENTS. Are you running this in a limited (non-GUI) environment?")}

	// initialize SDL3 window
	main_window := sdl3.CreateWindow("rinsedmeat", 1280, 720, sdl3.WindowFlags{})
	if (main_window == nil) {
		HaltSDLPrintingMessage(
			"Main window creation failed. The game cannot run without a window.",
		)
	}

	log.info("Engine shutdown complete!")
}
