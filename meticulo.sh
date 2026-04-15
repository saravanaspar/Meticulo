#!/bin/bash
# Meticulo launcher script
# Usage: ./meticulo.sh [command] [args...]
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 -m meticulo.cli "$@"
