# Meticulo

Zero-overhead local LLM runner powered by llama.cpp

> **Feel the raw power of your system.**

**Default port: 22434** (avoids Ollama conflict on 11434)

## Quick Install (Non-Coders)

1. **Download latest release** from: https://github.com/meticulo/meticulo/releases
2. Extract the ZIP file
3. Run the installer script:
   ```bash
   cd meticulo
   ./install.sh
   ```
4. Done! Start using:
   ```bash
   meticulo info
   ```

## Build from Source (Developers)

```bash
# Clone
git clone https://github.com/meticulo/meticulo.git
cd meticulo

# Install
pip install -e .

# Build llama.cpp engine
meticulo setup

# Verify
meticulo info
```

### Requirements
- Python 3.10+
- cmake, make, g++
- NVIDIA GPU + CUDA (optional)

## Commands Guide

```bash
# Pull models
meticulo pull llama3.2:1b              # From registry
meticulo pull user/repo:file.gguf      # From HuggingFace

# List models
meticulo list                          # Downloaded models
meticulo list --available              # All available
meticulo list --embeddings             # Embedding models

# Run model (interactive)
meticulo run llama3.2:1b

# Run with options
meticulo run llama3.2:1b --prompt "hello" --temperature 0.7

# API server (chat)
meticulo serve --model llama3.2:1b

# API server (embeddings)
meticulo serve-embed --model nomic-embed-text:v1.5-137m

# Status
meticulo ps                # Running servers + memory
meticulo info              # Hardware info

# Stop
meticulo stop

# Config
meticulo config show
meticulo config set auto_update true   # Auto-update from git (default: on)

# Delete model
meticulo rm llama3.2:1b
```

## API Usage

**OpenAI format (port 22434):**
```bash
curl http://localhost:22434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2:1b", "messages": [{"role": "user", "content": "Hi"}]}'
```

**Ollama format:**
```bash
curl http://localhost:22434/api/generate \
  -d '{"model": "llama3.2:1b", "prompt": "Hi"}'
```

**Embeddings (port 8081):**
```bash
curl http://localhost:8081/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "nomic-embed-text:v1.5-137m", "input": "Hello world"}'
```


## Status

### Completed
- ✅ Interactive chat (CLI)
- ✅ API server (OpenAI + Ollama compatible)
- ✅ Embedding server (separate port 8081)
- ✅ Model pulling from registry + HuggingFace
- ✅ Hardware detection + auto-tuning
- ✅ Python SDK
- ✅ TUI (Rust)
- ✅ Auto-update from git
- ✅ Config system

### In Progress
- 🔄 Tool calling / function calling
- 🔄 Structured output (JSON mode)
- 🔄 Vision/multimodal support

### Upcoming
- 📅 Windows/Linux system tray
- 📅 Claude/OpenAI SDK compatibility
- 📅 Model persistence (keep loaded)
- 📅 Streaming improvements

## Architecture

```
┌────────────┐     ┌────────────┐
│  Python    │────▶│   API      │────▶ llama-server
│  CLI/SDK   │     │  Server    │     (port 22435)
└────────────┘     └─────┬──────┘
                        │
┌────────────┐          │
│    TUI    │──────────┘
│  (Rust)   │
└────────────┘
```

## Configuration

```bash
# Data directory
export METICULO_HOME=~/.meticulo

# Models stored at: ~/.meticulo/models/
# Engine at: ~/.meticulo/engine/
# Config: ~/.meticulo/config.json
```

## License

MIT