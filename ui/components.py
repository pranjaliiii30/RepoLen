"""Small reusable Streamlit components."""

import base64
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from core.chunker import chunk_file, extract_python_symbols
from core.ingest import EXTENSIONS
from core.models import SourceFile

LANGUAGE_ALIASES = {
    "c_sharp": "csharp",
    "restructuredtext": "rst",
    "markdown": "markdown",
}


def display_language(language: str) -> str:
    """Map detected language names to Streamlit's code-fence names."""
    return LANGUAGE_ALIASES.get(language, language)


def render_copy_button(content: str) -> None:
    """Render an accessible clipboard button in a small isolated component."""
    encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
    components.html(
        f"""
        <button id="copy" style="background:#242d47;color:#e6e9f5;border:1px solid #35405f;
        border-radius:7px;padding:5px 10px;cursor:pointer;font:12px ui-monospace,monospace">
        Copy snippet</button>
        <script>
        const button = document.getElementById("copy");
        button.addEventListener("click", async () => {{
          try {{
            await navigator.clipboard.writeText(new TextDecoder().decode(
              Uint8Array.from(atob("{encoded}"), character => character.charCodeAt(0))
            ));
            button.textContent = "Copied";
            setTimeout(() => button.textContent = "Copy snippet", 1200);
          }} catch (error) {{
            button.textContent = "Copy unavailable";
          }}
        }});
        </script>
        """,
        height=38,
        scrolling=False,
    )


def render_sources(sources: list[Any]) -> None:
    """Show each retrieved code source with metadata and syntax highlighting."""
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        if not sources:
            st.caption("No code chunks were retrieved for this answer.")
        for result in sources:
            chunk = result.chunk
            st.markdown(
                f"**`{chunk.path}`** · `{chunk.symbol}` · lines "
                f"`{chunk.start_line}-{chunk.end_line}` · similarity `{result.similarity:.3f}`"
            )
            st.code(
                chunk.content,
                language=display_language(chunk.language),
                line_numbers=True,
                wrap_lines=True,
            )
            render_copy_button(chunk.content)


def _render_tree(
    node: dict[str, Any],
    prefix: str,
    on_select: Any,
    depth: int = 0,
) -> None:
    for name, value in sorted(node.items(), key=lambda pair: (not isinstance(pair[1], dict), pair[0].lower())):
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(value, dict):
            with st.expander(f"📁 {name}", expanded=depth == 0):
                _render_tree(value, path, on_select, depth + 1)
        elif st.button(f"📄 {name}", key=f"tree-{path}", use_container_width=True):
            on_select(path)


def render_file_tree(file_paths: list[str], on_select: Any) -> None:
    """Render a nested, collapsible file tree with selectable files."""
    tree: dict[str, Any] = {}
    for file_path in file_paths:
        parts = Path(file_path).parts
        node = tree
        for directory in parts[:-1]:
            node = node.setdefault(directory, {})
        node[parts[-1]] = file_path
    _render_tree(tree, "", on_select)


def render_file_contents(path: str, source_dir: Path) -> None:
    """Display a selected file and its extracted function/class symbols."""
    target = source_dir.joinpath(*Path(path).parts)
    try:
        resolved_root = source_dir.resolve()
        resolved_target = target.resolve()
        if resolved_root not in resolved_target.parents or not resolved_target.is_file():
            st.error("The selected file is not available in this indexed snapshot.")
            return
        content = resolved_target.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        st.error(f"Could not read the indexed file: {error}")
        return
    extension = Path(path).suffix.lower()
    detected_language = EXTENSIONS.get(extension)
    st.markdown(f"**`{path}`**")
    st.code(
        content,
        language=display_language(detected_language) if detected_language else "text",
        line_numbers=True,
        wrap_lines=True,
    )
    st.markdown("#### Extracted symbols")
    if detected_language:
        if detected_language == "python":
            symbols = extract_python_symbols(content)
        else:
            chunks = chunk_file(
                SourceFile(path=path, content=content, language=detected_language)
            )
            symbols = [
                (chunk.symbol, chunk.start_line)
                for chunk in chunks
                if " (lines " not in chunk.symbol
            ]
        if symbols:
            for name, line in symbols:
                st.markdown(f"- `{name}` — line {line}")
        else:
            st.caption("No supported functions or classes were found.")
    else:
        st.caption("Symbol extraction is unavailable for this file type.")


def render_mermaid_diagram(diagram: str) -> None:
    """Render a Mermaid diagram in an isolated browser component."""
    encoded = base64.b64encode(diagram.encode("utf-8")).decode("ascii")
    components.html(
        f"""
        <div id="mermaid-root" style="overflow:auto;background:#11182b;padding:12px;border-radius:12px"></div>
        <script type="module">
        import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
        const root = document.getElementById("mermaid-root");
        const graph = new TextDecoder().decode(
          Uint8Array.from(atob("{encoded}"), character => character.charCodeAt(0))
        );
        try {{
          mermaid.initialize({{ startOnLoad: false, theme: "dark" }});
          const rendered = await mermaid.render("repo-architecture", graph);
          root.innerHTML = rendered.svg;
        }} catch (error) {{
          root.textContent = graph;
        }}
        </script>
        """,
        height=480,
        scrolling=True,
    )
