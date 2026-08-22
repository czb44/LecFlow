from lecflow.llm.ollama import OllamaClient


def test_refine_notes_empty(monkeypatch):
    def fake_chat(model, messages):
        return {"message": {"content": "-"}}

    monkeypatch.setattr("lecflow.llm.ollama.ollama.chat", fake_chat)

    client = OllamaClient()

    result = client.refine_notes(["AUDIENCE:", "No.", "Get it?"])
    assert result == []


def test_refine_notes(monkeypatch):
    def fake_chat(model, messages):
        return {
            "message": {
                "content": (
                    "- Hash tables allow for constant-time lookup.\n- Keys must be immutable."
                )
            }
        }

    monkeypatch.setattr("lecflow.llm.ollama.ollama.chat", fake_chat)

    client = OllamaClient()

    result = client.refine_notes(
        [
            "Hash tables are helpful.",
            "No.",
        ]
    )

    assert result == [
        "Hash tables allow for constant-time lookup.",
        "Keys must be immutable.",
    ]


def test_filter_housekeeping_empty(monkeypatch):
    def fake_chat(model, messages):
        return {"message": {"content": "-"}}

    monkeypatch.setattr("lecflow.llm.ollama.ollama.chat", fake_chat)

    client = OllamaClient()

    result = client.filter_housekeeping(["AUDIENCE:", "No.", "Get it?"])
    assert result == []


def test_filter_housekeeping(monkeypatch):
    def fake_chat(model, messages):
        return {"message": {"content": ("- Class tomorrow is cancelled.\n- PSET 5 is Due Friday.")}}

    monkeypatch.setattr("lecflow.llm.ollama.ollama.chat", fake_chat)

    client = OllamaClient()

    result = client.filter_housekeeping(
        [
            "No class tommorrow.",
            "PSET 5 due Friday.",
        ]
    )

    assert result == [
        "Class tomorrow is cancelled.",
        "PSET 5 is Due Friday.",
    ]
