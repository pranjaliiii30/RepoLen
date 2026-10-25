"""Tests for Python AST-aware code chunking."""

from core.chunker import chunk_file, extract_python_symbols
from core.models import SourceFile


def source(content: str) -> SourceFile:
    return SourceFile(path="sample.py", content=content, language="python")


def test_chunks_functions_and_classes_with_line_numbers() -> None:
    content = "class Widget:\n    def render(self):\n        return 'ok'\n"
    chunks = chunk_file(source(content))

    assert [(chunk.symbol, chunk.start_line, chunk.end_line) for chunk in chunks] == [
        ("Widget", 1, 3),
        ("Widget.render", 2, 3),
    ]


def test_nested_classes_and_methods_are_qualified() -> None:
    content = (
        "class Outer:\n"
        "    class Inner:\n"
        "        def run(self):\n"
        "            pass\n"
    )
    chunks = chunk_file(source(content))

    assert [chunk.symbol for chunk in chunks] == [
        "Outer",
        "Outer.Inner",
        "Outer.Inner.run",
    ]


def test_decorators_are_included_in_function_chunk() -> None:
    content = "@classmethod\ndef create(cls):\n    return cls()\n"
    chunk = chunk_file(source(content))[0]

    assert chunk.symbol == "create"
    assert chunk.start_line == 1
    assert chunk.content.startswith("@classmethod")


def test_syntax_error_uses_line_fallback() -> None:
    content = "def broken(:\n    return 1\n"
    chunks = chunk_file(source(content), max_lines=1, overlap=0)

    assert len(chunks) == 2
    assert chunks[0].start_line == 1
    assert "def broken(" in chunks[0].content


def test_long_function_respects_chunk_limit_and_overlap() -> None:
    content = "def long_function():\n" + "".join(
        f"    value_{line} = {line}\n" for line in range(12)
    )
    chunks = chunk_file(source(content), max_lines=5, overlap=1)

    assert len(chunks) == 3
    assert all(chunk.end_line - chunk.start_line < 5 for chunk in chunks)
    assert chunks[0].symbol == "long_function (part 1)"
    assert chunks[1].start_line == chunks[0].end_line


def test_symbol_extraction_reports_nested_names() -> None:
    symbols = extract_python_symbols("class Box:\n    def open(self):\n        pass\n")

    assert symbols == [("Box", 1), ("Box.open", 2)]
