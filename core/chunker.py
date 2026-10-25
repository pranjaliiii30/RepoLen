"""Syntax-aware source chunking with safe line-based fallbacks."""

import ast
import logging
import re
from pathlib import PurePosixPath

from core.models import Chunk, SourceFile

logger = logging.getLogger(__name__)

_DEFINITION_TYPES = {
    "class_definition",
    "class_declaration",
    "function_definition",
    "function_declaration",
    "function_item",
    "method_definition",
    "method_declaration",
    "interface_declaration",
    "struct_item",
    "enum_item",
    "trait_item",
    "impl_item",
}


def _line_chunks(source: SourceFile, max_lines: int, overlap: int) -> list[Chunk]:
    lines = source.content.splitlines()
    if not lines:
        return []
    step = max(1, max_lines - overlap)
    chunks: list[Chunk] = []
    for offset in range(0, len(lines), step):
        section = lines[offset : offset + max_lines]
        if not section:
            continue
        chunks.append(
            Chunk(
                path=source.path,
                symbol=f"{PurePosixPath(source.path).name} (lines {offset + 1}-{offset + len(section)})",
                start_line=offset + 1,
                end_line=offset + len(section),
                content="\n".join(section),
                language=source.language,
            )
        )
        if offset + len(section) >= len(lines):
            break
    return chunks


def _symbol_chunks(
    source: SourceFile,
    symbol: str,
    start: int,
    end: int,
    lines: list[str],
    max_lines: int,
    overlap: int,
) -> list[Chunk]:
    """Keep long definitions within the configured chunk size."""
    step = max(1, max_lines - overlap)
    chunks: list[Chunk] = []
    part = 1
    offset = start
    while offset <= end:
        final = min(offset + max_lines - 1, end)
        label = symbol if start + max_lines > end else f"{symbol} (part {part})"
        chunks.append(
            Chunk(
                path=source.path,
                symbol=label,
                start_line=offset,
                end_line=final,
                content="\n".join(lines[offset - 1 : final]),
                language=source.language,
            )
        )
        if final == end:
            break
        offset += step
        part += 1
    return chunks


def _python_chunks(source: SourceFile, max_lines: int, overlap: int) -> list[Chunk]:
    try:
        tree = ast.parse(source.content)
    except SyntaxError:
        return _line_chunks(source, max_lines, overlap)

    lines = source.content.splitlines()
    chunks: list[Chunk] = []

    class DefinitionVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.parents: list[str] = []

        def _visit_definition(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            name = node.name
            qualified_name = ".".join((*self.parents, name))
            decorator_lines = [
                decorator.lineno for decorator in getattr(node, "decorator_list", [])
            ]
            start = min([node.lineno, *decorator_lines])
            end = getattr(node, "end_lineno", node.lineno)
            chunks.extend(
                _symbol_chunks(
                    source, qualified_name, start, end, lines, max_lines, overlap
                )
            )
            self.parents.append(name)
            self.generic_visit(node)
            self.parents.pop()

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self._visit_definition(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_definition(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_definition(node)

    DefinitionVisitor().visit(tree)
    if chunks:
        return chunks
    return _line_chunks(source, max_lines, overlap)


def _tree_sitter_chunks(source: SourceFile, max_lines: int, overlap: int) -> list[Chunk]:
    try:
        from tree_sitter_language_pack import get_parser

        parser = get_parser(source.language)
        tree = parser.parse(source.content.encode("utf-8"))
    except (ImportError, LookupError, ValueError):
        return _line_chunks(source, max_lines, overlap)
    except Exception as error:
        logger.warning("Tree-sitter parsing failed for %s; using line chunks: %s", source.path, error)
        return _line_chunks(source, max_lines, overlap)

    lines = source.content.splitlines()
    chunks: list[Chunk] = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type in _DEFINITION_TYPES:
            start = node.start_point.row
            end = node.end_point.row
            if end >= len(lines):
                end = len(lines) - 1
            if start <= end and lines:
                raw_name = node.child_by_field_name("name")
                name = (
                    source.content.encode("utf-8")[raw_name.start_byte : raw_name.end_byte].decode(
                        "utf-8", errors="replace"
                    )
                    if raw_name is not None
                    else node.type
                )
                chunks.extend(
                    _symbol_chunks(
                        source,
                        name,
                        start + 1,
                        end + 1,
                        lines,
                        max_lines,
                        overlap,
                    )
                )
        stack.extend(reversed(node.children))
    if chunks:
        return sorted(chunks, key=lambda chunk: (chunk.start_line, chunk.end_line))
    return _line_chunks(source, max_lines, overlap)


def chunk_file(
    source: SourceFile, max_lines: int = 100, overlap: int = 15
) -> list[Chunk]:
    """Split a source file into symbol-aware chunks, falling back safely."""
    if max_lines < 1 or overlap < 0 or overlap >= max_lines:
        raise ValueError("max_lines must be positive and overlap must be smaller than max_lines")
    if source.language == "python":
        return _python_chunks(source, max_lines, overlap)
    return _tree_sitter_chunks(source, max_lines, overlap)


def extract_python_symbols(content: str) -> list[tuple[str, int]]:
    """Return function and class names with their starting source lines."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        pattern = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)", re.MULTILINE)
        return [
            (match.group(1), content.count("\n", 0, match.start()) + 1)
            for match in pattern.finditer(content)
        ]

    symbols: list[tuple[str, int]] = []

    class SymbolVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.parents: list[str] = []

        def _visit_definition(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            symbols.append((".".join((*self.parents, node.name)), node.lineno))
            self.parents.append(node.name)
            self.generic_visit(node)
            self.parents.pop()

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self._visit_definition(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_definition(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_definition(node)

    SymbolVisitor().visit(tree)
    return symbols
