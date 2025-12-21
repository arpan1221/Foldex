import pytest

from app.rag.stream_sanitizer import ThinkTagStreamFilter


def test_stream_filter_passes_through_normal_text() -> None:
    f = ThinkTagStreamFilter(preface_buffer_limit=0)
    assert f.process("Hello") == "Hello"
    assert f.process(" world") == " world"


def test_stream_filter_strips_think_block_even_when_split_across_tokens() -> None:
    f = ThinkTagStreamFilter(preface_buffer_limit=1024)

    # Preamble + split open tag + split close tag + answer
    out = []
    out.append(f.process("Hmm, the user wants to see reasoning. "))
    out.append(f.process("<th"))
    out.append(f.process("ink>SECRET"))
    out.append(f.process(" STUFF</th"))
    out.append(f.process("ink>Final answer."))

    # Should not leak preface or think content; should include final answer.
    assert "".join(out) == "Final answer."


def test_stream_filter_drops_unclosed_think_content() -> None:
    f = ThinkTagStreamFilter(preface_buffer_limit=1024)
    assert f.process("<think>should never show") == ""
    assert f.process(" more thinking") == ""


@pytest.mark.parametrize(
    "open_tag,close_tag",
    [
        ("<think>", "</think>"),
        ("<antml:thinking>", "</antml:thinking>"),
    ],
)
def test_stream_filter_supports_multiple_tag_variants(open_tag: str, close_tag: str) -> None:
    f = ThinkTagStreamFilter(preface_buffer_limit=1024)
    out = []
    out.append(f.process(f"pre {open_tag}secret"))
    out.append(f.process(f" hidden{close_tag} post"))
    assert "".join(out) == " post"


