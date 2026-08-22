from typing import Protocol


class LLMClient(Protocol):
    """Structure for LLM providers.
    Allows for abillity to add other providers (OpenAI, etc.) later"""

    def refine_notes(self, notes_block: str) -> str:
        """Refines a block of notes, removing  insignificant sentences
        and fixing rule-based unit-type classification errors."""
        ...
