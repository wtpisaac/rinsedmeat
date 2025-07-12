#!/bin/bash

THIS_DIR=$(dirname "$BASH_SOURCE")

glslangValidator -V -S vert -o "$THIS_DIR/shaders/shader.vert.spv" "$THIS_DIR/shaders/shader.vert"
glslangValidator -V -S frag -o "$THIS_DIR/shaders/shader.frag.spv" "$THIS_DIR/shaders/shader.frag"

