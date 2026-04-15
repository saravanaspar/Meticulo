#!/bin/bash
set -e

echo "╔══════════════════════════════════════╗"
echo "║    Meticulo TUI - Build Script       ║"
echo "╚══════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

build_target() {
    local target="$1"
    local label="$2"
    echo "▸ Building for $label ($target)..."
    if rustup target list --installed | grep -q "$target"; then
        cargo build --release --target "$target" 2>&1 | tail -1
        local bin="target/$target/release/meticulo-tui"
        if [ -f "$bin" ]; then
            local size=$(du -h "$bin" | cut -f1)
            echo "  ✓ $bin ($size)"
        elif [ -f "${bin}.exe" ]; then
            local size=$(du -h "${bin}.exe" | cut -f1)
            echo "  ✓ ${bin}.exe ($size)"
        fi
    else
        echo "  ⚠ Target not installed. Run: rustup target add $target"
    fi
    echo ""
}

case "${1:-native}" in
    native)
        echo "▸ Building for current platform..."
        cargo build --release
        bin="target/release/meticulo-tui"
        size=$(du -h "$bin" | cut -f1)
        echo "  ✓ $bin ($size)"
        ;;
    linux)
        build_target "x86_64-unknown-linux-gnu" "Linux x86_64"
        build_target "aarch64-unknown-linux-gnu" "Linux ARM64 (RPi 4/5)"
        build_target "armv7-unknown-linux-gnueabihf" "Linux ARMv7 (RPi 3)"
        ;;
    windows)
        build_target "x86_64-pc-windows-gnu" "Windows x86_64"
        build_target "i686-pc-windows-gnu" "Windows x86 (32-bit)"
        build_target "aarch64-pc-windows-msvc" "Windows ARM64"
        ;;
    all)
        echo "Building all targets..."
        echo ""
        build_target "x86_64-unknown-linux-gnu" "Linux x86_64"
        build_target "aarch64-unknown-linux-gnu" "Linux ARM64 (RPi 4/5)"
        build_target "armv7-unknown-linux-gnueabihf" "Linux ARMv7 (RPi 3)"
        build_target "x86_64-pc-windows-gnu" "Windows x86_64"
        ;;
    *)
        echo "Usage: $0 [native|linux|windows|all]"
        exit 1
        ;;
esac

echo ""
echo "Done."
