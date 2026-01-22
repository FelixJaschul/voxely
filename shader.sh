#!/bin/bash
set -euo pipefail

SHADER_DIR="shaders"
OUTPUT_DIR="shaders"

mkdir -p "$OUTPUT_DIR"

compile_one () {
    local src="$1"
    local ext="${src##*.}"
    local base="$(basename "$src")"
    local name="${base%.*}"

    local stage=""
    case "$ext" in
        comp) stage="comp" ;;
        vert) stage="vert" ;;
        frag) stage="frag" ;;
        *)
            echo "Skipping $src (unknown extension: $ext)"
            return
            ;;
    esac

    local spv="$OUTPUT_DIR/$base.spv"
    local msl="$OUTPUT_DIR/$base.metal"

    echo "glslc ($stage): $src -> $spv"
    glslc -fshader-stage="$stage" "$src" -o "$spv"

    echo "spirv-cross ($stage): $spv -> $msl"
    # Force stage selection, and rename entry point to main0 so your C++ can use entrypoint="main0".
    spirv-cross "$spv" \
        --msl --stage "$stage" \
        --rename-entry-point main main0 "$stage" \
        --output "$msl"
}

shopt -s nullglob
for f in "$SHADER_DIR"/*.comp "$SHADER_DIR"/*.vert "$SHADER_DIR"/*.frag; do
    compile_one "$f"
done

echo "✓ Done. Outputs written next to sources in: $OUTPUT_DIR"
