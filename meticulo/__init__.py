"""Meticulo - Zero-overhead local LLM runner powered by llama.cpp"""

__version__ = "0.1.0"

from .sdk import ChatMessage, MeticuloAPIError, MeticuloClient, ModelInfo

__all__ = [
	"__version__",
	"MeticuloClient",
	"MeticuloAPIError",
	"ChatMessage",
	"ModelInfo",
]
