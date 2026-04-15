"""Types for the Meticulo Python SDK."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChatMessage:
    """Single chat message."""

    role: str
    content: str


@dataclass
class ModelInfo:
    """Model metadata returned by the API."""

    id: str
    object: str = "model"
    owned_by: str = "local"
    size: int = 0
    size_human: str = ""


@dataclass
class ChatChoice:
    """Chat completion choice."""

    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


@dataclass
class ChatCompletionResponse:
    """Chat completion response."""

    id: str
    object: str
    created: int
    model: str
    choices: List[ChatChoice] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingData:
    """Single embedding vector payload."""

    index: int
    embedding: List[float]
    object: str = "embedding"


@dataclass
class EmbeddingsResponse:
    """Embeddings API response."""

    object: str
    data: List[EmbeddingData]
    model: str
    usage: Dict[str, Any] = field(default_factory=dict)
