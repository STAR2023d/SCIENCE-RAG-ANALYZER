from science_rag.models import ImageHit, Retrieval
from science_rag.prompts import SYSTEM_PROMPT, build_messages, build_user_prompt


def test_prompt_contains_context_and_question():
    retrieval = Retrieval(
        contexts=["Plants make glucose.", "Chlorophyll is green."], sources=["a.pdf"]
    )
    prompt = build_user_prompt("What do plants make?", retrieval)
    assert "- Plants make glucose." in prompt
    assert "- Chlorophyll is green." in prompt
    assert "Question: What do plants make?" in prompt
    assert "Related figures" not in prompt


def test_prompt_lists_figures_when_present():
    figure = ImageHit(
        path="/x.png", caption="A diagram of a leaf", source="a.pdf", page=4, score=0.8
    )
    prompt = build_user_prompt("q", Retrieval(contexts=["c"], sources=[], images=[figure]))
    assert "Related figures" in prompt
    assert "a.pdf, page 4" in prompt
    assert "A diagram of a leaf" in prompt


def test_standalone_images_have_no_page_number():
    figure = ImageHit(path="/x.png", caption="cap", source="pic.png", page=0, score=0.8)
    prompt = build_user_prompt("q", Retrieval(contexts=["c"], sources=[], images=[figure]))
    assert "pic.png]" in prompt and "page" not in prompt.split("Related figures")[1]


def test_messages_shape():
    messages = build_messages("q", Retrieval(contexts=["c"], sources=[]))
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
