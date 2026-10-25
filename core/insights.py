"""Repository statistics and import-based architecture summaries."""

import ast
import re
from collections import Counter
from pathlib import Path, PurePosixPath

from core.models import SourceFile


def repository_stats(files: list[SourceFile], chunk_count: int) -> dict[str, object]:
    """Compute file, line, language, size, and Python complexity statistics."""
    language_counts = Counter(source.language for source in files)
    sizes = [(source.path, len(source.content.splitlines())) for source in files]
    complexity = [
        (source.path, len(_python_functions(source.content)))
        for source in files
        if source.language == "python"
    ]
    return {
        "file_count": len(files),
        "line_count": sum(lines for _, lines in sizes),
        "chunk_count": chunk_count,
        "language_counts": dict(language_counts),
        "largest_files": sorted(sizes, key=lambda item: item[1], reverse=True)[:10],
        "complex_files": sorted(complexity, key=lambda item: item[1], reverse=True)[:10],
    }


def _python_functions(content: str) -> list[str]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []
    return [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def import_edges(files: list[SourceFile]) -> list[tuple[str, str]]:
    """Extract local Python and JavaScript/TypeScript module import edges."""
    paths = {PurePosixPath(source.path).with_suffix("").as_posix(): source.path for source in files}
    edges: set[tuple[str, str]] = set()
    js_import = re.compile(
        r"""(?:from\s+['"]([^'"]+)['"]|import\s*(?:[^'"]*?\sfrom\s*)?['"]([^'"]+)['"])"""
    )
    for source in files:
        source_module = PurePosixPath(source.path).with_suffix("").as_posix()
        if source.language == "python":
            try:
                tree = ast.parse(source.content)
            except SyntaxError:
                continue
            modules = [
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module
            ]
            modules.extend(
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            )
            for module in modules:
                target = module.replace(".", "/")
                for candidate in (target, f"{target}/__init__", f"{target}/index"):
                    if candidate in paths:
                        edges.add((source.path, paths[candidate]))
                        break
        elif source.language in {"javascript", "typescript"}:
            for match in js_import.finditer(source.content):
                module = match.group(1) or match.group(2) or ""
                if not module.startswith("."):
                    continue
                base = PurePosixPath(source.path).parent.joinpath(module)
                candidates = [base.as_posix()]
                candidates.extend(
                    f"{base.as_posix()}{suffix}"
                    for suffix in (".js", ".jsx", ".ts", ".tsx")
                )
                candidates.append(f"{base.as_posix()}/index")
                for candidate in candidates:
                    normalized = str(Path(candidate)).replace("\\", "/")
                    if normalized in paths:
                        edges.add((source.path, paths[normalized]))
                        break
    return sorted(edges)


def mermaid_diagram(files: list[SourceFile], limit: int = 40) -> str:
    """Create a Mermaid graph from local import relationships."""
    edges = import_edges(files)[:limit]
    if not edges:
        return "graph TD\n  A[No local import edges detected]"
    nodes: dict[str, str] = {}
    lines = ["graph TD"]
    for source, target in edges:
        for path in (source, target):
            if path not in nodes:
                node_id = f"n{len(nodes)}"
                nodes[path] = node_id
                label = path.replace('"', "'")
                lines.append(f'  {node_id}["{label}"]')
        lines.append(f"  {nodes[source]} --> {nodes[target]}")
    return "\n".join(lines)


def overview_prompt(files: list[SourceFile], stats: dict[str, object]) -> str:
    """Build an evidence-only prompt for an indexed repository overview."""
    excerpts = []
    priority_files = [
        source
        for source in files
        if PurePosixPath(source.path).name.lower()
        in {"readme.md", "pyproject.toml", "package.json", "requirements.txt", "go.mod", "cargo.toml"}
    ]
    priority_files.extend(source for source in files if source not in priority_files)
    for source in priority_files[:12]:
        excerpt = "\n".join(
            f"{line_number}: {line}"
            for line_number, line in enumerate(source.content.splitlines()[:60], start=1)
        )
        excerpts.append(f"--- {source.path} ---\n{excerpt}")
    return (
        f"Repository statistics: {stats}\n\n"
        "Indexed source excerpts (the only evidence you may use):\n"
        + "\n\n".join(excerpts)
    )


OVERVIEW_SYSTEM_PROMPT = """Write a concise, developer-friendly repository overview with sections for purpose, tech stack, folder structure, main entry points, and how to run it. Use only the supplied indexed code excerpts. Cite every factual claim using [file_path:start_line-end_line], matching the line numbers shown in the excerpts. For any item the excerpts do not support, say exactly "I couldn't find this in the indexed code". Never invent commands, files, or symbols."""
