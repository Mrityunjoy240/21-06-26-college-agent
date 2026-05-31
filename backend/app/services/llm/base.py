from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any


class BaseLLM(ABC):
    """Abstract base class for LLM clients"""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 500,
        stream: bool = True,
    ) -> AsyncGenerator[str, None]:
        """Stream chat response as tokens"""
        pass

    @abstractmethod
    async def chat_complete(
        self, messages: list[dict[str, str]], temperature: float = 0.0, max_tokens: int = 500
    ) -> str:
        """Get complete chat response"""
        pass

    @abstractmethod
    async def structured_output(
        self, messages: list[dict[str, str]], schema: dict[str, Any], temperature: float = 0.0
    ) -> dict[str, Any]:
        """Get structured JSON output"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if LLM is available"""
        pass
