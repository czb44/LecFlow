import ollama

from .prompts import NOTE_REFINEMENT_PROMPT


class OllamaClient:
    def __init__(self, model: str = "qwen3:4b-instruct-2507-q4_K_M"):
        # Fast, lightweight "qwen3:4b-instruct-2507-q4_K_M" default
        self.model = model

    def refine_notes(self, sentences_block: list[str]) -> list[str]:
        """Refines a block of lecture sentences with Ollama model.
        Removes insignificant or filler sentences"""
        # Ollama requires content to be a string
        content = "\n".join(f"- {sentence}" for sentence in sentences_block)

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": NOTE_REFINEMENT_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": content,
                    },
                ],
            )
        except Exception:
            raise RuntimeError(
                "Ollama refinement failed. "
                "Make sure Ollama is correctly running with correct model installed."
            )

        result = response["message"]["content"]

        blank_outputs = ["", "-", " ", "- ", " -", " - "]
        if result.strip() in blank_outputs:
            return []
        else:
            return [line.removeprefix("- ").strip() for line in result.splitlines() if line.strip()]


if __name__ == "__main__":
    client = OllamaClient()

    sentences_block = [
        "So what we're basically saying here is that merge sort takes O(n log n) time.",
        "Pretty cool, right?",
        "The important thing is that comparison sorting has an Omega(n log n) lower bound.",
        "I think we talked about this last week.",
        "Therefore merge sort is asymptotically optimal in the comparison model.",
    ]

    result = client.refine_notes(sentences_block)

    print(result)
