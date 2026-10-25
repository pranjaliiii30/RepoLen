"""RepoLens: local code indexing and grounded repository Q&A."""

import json
import logging
import os
import re
import uuid
from collections.abc import Callable
from collections import deque
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from core.chunker import chunk_file
from core.ingest import (
    clone_repository,
    collect_source_files,
    language_counts,
    persist_source_files,
    remove_snapshot,
    repository_id,
    repository_name,
    sanitize_repository_url,
)
from core.insights import (
    OVERVIEW_SYSTEM_PROMPT,
    mermaid_diagram,
    overview_prompt,
    repository_stats,
)
from core.indexer import create_collection
from core.llm import (
    create_gemini_client,
    generate_followups,
    generate_text,
    rewrite_query,
    stream_answer,
)
from core.models import RepoIndex, SourceFile
from core.retriever import hybrid_search
from ui.components import (
    render_file_contents,
    render_file_tree,
    render_mermaid_diagram,
    render_sources,
)
from ui.styles import apply_styles

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("repolens")
debug_messages: deque[str] = deque(maxlen=100)


class BufferLogHandler(logging.Handler):
    """Keep a bounded in-memory copy of application diagnostics for the sidebar."""

    def emit(self, record: logging.LogRecord) -> None:
        debug_messages.append(self.format(record))


buffer_handler = BufferLogHandler()
buffer_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
logging.getLogger().addHandler(buffer_handler)
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
REGISTRY_PATH = DATA_DIR / "repos.json"
INDEX_DIR = DATA_DIR / "chroma"
SOURCE_DIR = DATA_DIR / "sources"


@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str) -> SentenceTransformer:
    """Load and cache the selected local sentence-transformer model."""
    return SentenceTransformer(model_name)


@st.cache_resource(show_spinner=False)
def load_chroma_client(path: str):
    """Load and cache a persistent Chroma client."""
    import chromadb

    return chromadb.PersistentClient(path=path)


def load_registry() -> dict[str, RepoIndex]:
    """Load indexed repository metadata from the local JSON registry."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_PATH.exists():
        return {}
    try:
        raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        return {key: RepoIndex.from_dict(value) for key, value in raw.items()}
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        logger.exception("Repository registry could not be loaded")
        st.error(f"Could not read the repository registry: {error}")
        return {}


def save_registry(registry: dict[str, RepoIndex]) -> None:
    """Atomically persist repository metadata."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary = REGISTRY_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({key: item.to_dict() for key, item in registry.items()}, indent=2),
        encoding="utf-8",
    )
    temporary.replace(REGISTRY_PATH)


def index_repository(
    url: str,
    branch: str,
    token: str,
    embedder: SentenceTransformer,
    client,
    progress_callback: Callable[[int, str], None] | None = None,
) -> RepoIndex:
    """Clone, chunk, snapshot, and index a repository with a new collection."""
    repo_id = repository_id(url, branch or "HEAD")
    version = uuid.uuid4().hex[:10]
    collection_name = f"repo_{repo_id}_{version}"
    snapshot = SOURCE_DIR / repo_id / version
    files = []
    chunks = []
    created_collection = False
    try:
        with clone_repository(url, branch, token or None, progress_callback) as checkout:
            files = collect_source_files(checkout)
            if not files:
                raise ValueError("No supported text source files were found in that repository.")
            if progress_callback:
                progress_callback(35, f"Found {len(files)} supported source files.")
            for source in files:
                chunks.extend(chunk_file(source))
            if not chunks:
                raise ValueError("No indexable text chunks were found in the repository.")
            if progress_callback:
                progress_callback(55, f"Extracted {len(chunks)} code-aware chunks.")
            snapshot.mkdir(parents=True, exist_ok=False)
            persist_source_files(files, snapshot)
        if progress_callback:
            progress_callback(70, "Creating embeddings and persistent search index.")
        create_collection(client, collection_name, chunks, embedder)
        created_collection = True
        if progress_callback:
            progress_callback(95, "Saving repository metadata.")
        return RepoIndex(
            repo_id=repo_id,
            name=repository_name(url),
            url=sanitize_repository_url(url),
            branch=branch or "HEAD",
            collection_name=collection_name,
            source_dir=str(snapshot.relative_to(APP_DIR)),
            indexed_at=datetime.now(timezone.utc).isoformat(),
            file_count=len(files),
            line_count=sum(len(source.content.splitlines()) for source in files),
            chunk_count=len(chunks),
            language_counts=language_counts(files),
        )
    except Exception:
        if created_collection:
            client.delete_collection(name=collection_name)
        if snapshot.exists():
            remove_snapshot(snapshot, SOURCE_DIR)
        raise


def delete_repo(repo: RepoIndex, registry: dict[str, RepoIndex], chroma_client) -> None:
    """Remove one registered index and its source snapshot safely."""
    chroma_client.delete_collection(name=repo.collection_name)
    source = (APP_DIR / repo.source_dir).resolve()
    remove_snapshot(source, SOURCE_DIR)
    del registry[repo.repo_id]
    save_registry(registry)


def source_files_for(repo: RepoIndex) -> list[SourceFile]:
    """Load the indexed file snapshot for repository views."""
    base = (APP_DIR / repo.source_dir).resolve()
    if SOURCE_DIR.resolve() not in base.parents or not base.is_dir():
        raise FileNotFoundError("The indexed source snapshot is missing or invalid.")
    return collect_source_files(base)


def show_hero() -> None:
    """Render the application name and concise tagline."""
    st.markdown(
        '<div class="hero"><div class="eyebrow">Local-first code intelligence</div>'
        "<h1>🔎 RepoLens</h1>"
        "<p>Ask grounded questions, explore source, and understand a repository faster.</p></div>",
        unsafe_allow_html=True,
    )


def render_citation_badges(answer: str) -> None:
    """Render cited file ranges as compact, safely escaped badges."""
    citations = list(dict.fromkeys(re.findall(r"\[([^\[\]]+:\d+-\d+)\]", answer)))
    if citations:
        import html

        badges = " ".join(
            f'<span class="citation-badge">{html.escape(citation)}</span>'
            for citation in citations
        )
        st.markdown(badges, unsafe_allow_html=True)


def sidebar(registry: dict[str, RepoIndex], chroma_client):
    """Render repository, model, filter, and diagnostics controls."""
    with st.sidebar:
        st.markdown("## Workspace")
        with st.form("repository-index-form"):
            url = st.text_input("Repository URL", placeholder="https://github.com/owner/repo")
            branch = st.text_input("Branch (optional)")
            token = st.text_input("Access token (private repos)", type="password")
            index_clicked = st.form_submit_button("Index repository", type="primary", use_container_width=True)
        st.divider()
        st.markdown("### Model settings")
        gemini_model = st.selectbox(
            "Gemini model",
            ["gemini-2.5-flash", "gemini-2.0-flash"],
            key="gemini_model",
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        top_k = st.slider("Retrieved chunks", 2, 12, 6)
        embedding_model = st.selectbox(
            "Embedding model",
            [
                "sentence-transformers/all-MiniLM-L6-v2",
                "BAAI/bge-small-en-v1.5",
            ],
        )
        st.divider()
        st.markdown("### Filters")
        active = [repo for repo in registry.values() if repo.active]
        selected_repo_id = st.selectbox(
            "Active repository",
            options=[repo.repo_id for repo in active],
            format_func=lambda key: f"{registry[key].name} ({registry[key].branch})",
            index=0 if active else None,
            placeholder="No indexed repositories",
        )
        repo = registry.get(selected_repo_id) if selected_repo_id else None
        languages = sorted(repo.language_counts) if repo else []
        language = st.selectbox("Language", ["All languages", *languages])
        folders = ["All folders"]
        if repo:
            try:
                folders.extend(
                    sorted(
                        {
                            PurePosixPath(source.path).parent.as_posix() or "(root)"
                            for source in source_files_for(repo)
                        }
                    )
                )
            except (OSError, ValueError, FileNotFoundError) as error:
                st.warning(f"Could not load repository folders: {error}")
        folder = st.selectbox("Folder", folders)
        if active:
            st.markdown("### Indexed repositories")
            for item in list(registry.values()):
                col1, col2 = st.columns([3, 1])
                with col1:
                    is_active = st.checkbox(
                        item.name,
                        value=item.active,
                        key=f"active-{item.repo_id}",
                    )
                with col2:
                    if st.button("Delete", key=f"delete-{item.repo_id}", help="Delete this local index"):
                        try:
                            delete_repo(item, registry, chroma_client)
                            if st.session_state.get("active_repo") == item.repo_id:
                                st.session_state.pop("active_repo", None)
                            st.rerun()
                        except Exception as error:
                            logger.exception("Could not delete repository index")
                            st.error(f"Could not delete this index: {error}")
                if is_active != item.active:
                    item.active = is_active
                    save_registry(registry)
        if st.button("Clear chat", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()
        with st.expander("Debug log"):
            st.caption(f"Latest {len(debug_messages)} application log messages")
            st.code("\n".join(debug_messages) or "No diagnostics yet.", language="text", wrap_lines=True)
        return {
            "url": url,
            "branch": branch,
            "token": token,
            "index_clicked": index_clicked,
            "repo": repo,
            "language": None if language == "All languages" else language,
            "folder": None if folder == "All folders" else folder,
            "gemini_model": gemini_model,
            "temperature": temperature,
            "top_k": top_k,
            "embedding_model": embedding_model,
        }


def show_onboarding() -> None:
    """Display first-run instructions and safe public repository examples."""
    st.markdown(
        """
        <div class="card">
        <h3>Start exploring a repository</h3>
        <ol>
          <li>Paste a public Git URL in the sidebar (or add a token for a private repository).</li>
          <li>Choose a branch and select <b>Index repository</b>.</li>
          <li>Ask a question in Chat, or browse files and insights in the other tabs.</li>
        </ol>
        <p>Try: <code>https://github.com/pallets/flask</code>,
        <code>https://github.com/psf/requests</code>, or
        <code>https://github.com/streamlit/streamlit</code>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat(
    repo: RepoIndex | None,
    settings: dict,
    chroma_client,
    embedder: SentenceTransformer | None,
) -> None:
    """Render the grounded chat, streaming answers and source panels."""
    actions = [
        "Explain this repo",
        "Show architecture overview",
        "Find entry points",
        "List API routes",
        "Where is error handling?",
        "Where are the tests?",
    ]
    columns = st.columns(3)
    for index, action in enumerate(actions):
        if columns[index % 3].button(action, key=f"quick-{index}", use_container_width=True):
            st.session_state["pending_question"] = action
            st.rerun()

    if repo is None or embedder is None:
        show_onboarding()
    for message_index, message in enumerate(st.session_state.get("chat_history", [])):
        with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_citation_badges(message["content"])
                render_sources(message.get("sources", []))
                suggestions = message.get("suggestions", [])
                if suggestions:
                    st.caption("Suggested follow-ups")
                    cols = st.columns(len(suggestions))
                    for index, suggestion in enumerate(suggestions):
                        if cols[index].button(
                            suggestion,
                            key=f"history-followup-{message_index}-{index}",
                        ):
                            st.session_state["pending_question"] = suggestion
                            st.rerun()

    prompt = st.chat_input("Ask a question about the indexed code...", disabled=repo is None)
    pending = st.session_state.pop("pending_question", None)
    prompt = pending or prompt
    if not prompt:
        return
    st.session_state.setdefault("chat_history", []).append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
    sources = []
    with st.chat_message("assistant", avatar="🤖"):
        if repo is None or embedder is None:
            st.warning("Index a repository and load the embedding model before asking questions.")
            answer = "I couldn't find this in the indexed code"
        else:
            try:
                collection = chroma_client.get_collection(repo.collection_name)
                gemini = create_gemini_client()
                with st.status("Searching indexed code and preparing an answer...", expanded=False):
                    def safe_rewrite(question: str) -> list[str]:
                        try:
                            return rewrite_query(
                                gemini, settings["gemini_model"], question
                            )
                        except Exception as error:
                            logger.warning("Query rewrite unavailable; using the original query: %s", error)
                            return []

                    sources = hybrid_search(
                        collection,
                        prompt,
                        embedder,
                        top_k=settings["top_k"],
                        language=settings["language"],
                        folder=settings["folder"],
                        query_rewriter=safe_rewrite,
                    )
                answer = st.write_stream(
                    stream_answer(
                        gemini,
                        settings["gemini_model"],
                        prompt,
                        sources,
                        temperature=settings["temperature"],
                    )
                )
            except Exception as error:
                logger.exception("Chat answer failed")
                answer = "I couldn't find this in the indexed code"
                st.error(f"Could not answer this question: {error}")
                st.markdown(answer)
        render_citation_badges(answer)
        render_sources(sources)
        suggestions = []
        followup_clicked = False
        if repo and answer:
            try:
                suggestions = generate_followups(
                    create_gemini_client(),
                    settings["gemini_model"],
                    prompt,
                    answer,
                )
            except Exception as error:
                logger.warning("Follow-up suggestion generation failed: %s", error)
                st.caption(f"Generated follow-ups are unavailable; showing starter questions: {error}")
                suggestions = [
                    "Which tests cover this behavior?",
                    "What other files are involved?",
                    "Where is this called from?",
                ]
            if suggestions:
                st.caption("Suggested follow-ups")
                cols = st.columns(len(suggestions))
                for index, suggestion in enumerate(suggestions):
                    if cols[index].button(
                        suggestion,
                        key=f"new-followup-{len(st.session_state['chat_history'])}-{index}",
                    ):
                        st.session_state["pending_question"] = suggestion
                        followup_clicked = True
    st.session_state["chat_history"].append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "suggestions": suggestions,
        }
    )
    if followup_clicked:
        st.rerun()


def render_explorer(repo: RepoIndex | None) -> None:
    """Render repository file tree and source viewer."""
    if repo is None:
        st.info("Index a repository to browse its files.")
        return
    try:
        source_dir = APP_DIR / repo.source_dir
        files = source_files_for(repo)
    except (OSError, ValueError, FileNotFoundError) as error:
        st.error(f"Could not load this repository snapshot: {error}")
        return
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown("#### Files")
        render_file_tree(
            [source.path for source in files],
            lambda path: st.session_state.__setitem__("selected_file", path),
        )
    with right:
        selected = st.session_state.get("selected_file")
        file_paths = {source.path for source in files}
        if selected not in file_paths:
            selected = files[0].path if files else None
        if selected:
            render_file_contents(selected, source_dir)
        else:
            st.info("No displayable files were found.")


def render_insights(repo: RepoIndex | None, settings: dict) -> None:
    """Render repository statistics, language chart, overview, and module graph."""
    if repo is None:
        st.info("Index a repository to view insights.")
        return
    try:
        files = source_files_for(repo)
    except (OSError, ValueError, FileNotFoundError) as error:
        st.error(f"Could not load repository insights: {error}")
        return
    stats = repository_stats(files, repo.chunk_count)
    metrics = st.columns(3)
    metrics[0].metric("Files", stats["file_count"])
    metrics[1].metric("Lines of code", f'{stats["line_count"]:,}')
    metrics[2].metric("Chunks", stats["chunk_count"])
    left, right = st.columns(2)
    with left:
        st.markdown("#### Language breakdown")
        counts = stats["language_counts"]
        if counts:
            figure = px.pie(
                names=list(counts.keys()),
                values=list(counts.values()),
                hole=0.58,
                color_discrete_sequence=["#8b80f9", "#5f87e8", "#43b6a2", "#e7a75b", "#d9749a"],
            )
            figure.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#dce2f4",
                margin=dict(l=0, r=0, t=10, b=10),
                legend_title_text="Language",
            )
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.caption("No languages detected.")
    with right:
        st.markdown("#### Largest files")
        st.dataframe(
            [{"File": path, "Lines": lines} for path, lines in stats["largest_files"]],
            hide_index=True,
            use_container_width=True,
        )
        st.markdown("#### Most complex Python files")
        st.dataframe(
            [{"File": path, "Function count": count} for path, count in stats["complex_files"]],
            hide_index=True,
            use_container_width=True,
        )
    st.markdown("#### Repo Overview")
    if repo.overview:
        st.markdown(repo.overview)
    elif st.button("Generate Repo Overview", type="primary"):
        try:
            client = create_gemini_client()
            with st.status("Generating an overview from indexed files...", expanded=False):
                overview = generate_text(
                    client,
                    settings["gemini_model"],
                    overview_prompt(files, stats),
                    OVERVIEW_SYSTEM_PROMPT,
                    temperature=0.2,
                )
            repo.overview = overview
            registry = load_registry()
            registry[repo.repo_id] = repo
            save_registry(registry)
            st.rerun()
        except Exception as error:
            logger.exception("Overview generation failed")
            st.error(f"Could not generate the repository overview: {error}")
    else:
        st.caption("Generate a grounded summary of the repository purpose, stack, folders, entry points, and run instructions.")
    with st.expander("Module / dependency diagram"):
        render_mermaid_diagram(mermaid_diagram(files))


def main() -> None:
    """Run the RepoLens Streamlit application."""
    apply_styles()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    registry = load_registry()
    chroma_client = load_chroma_client(str(INDEX_DIR))
    settings = sidebar(registry, chroma_client)
    if settings["index_clicked"]:
        if not settings["url"].strip():
            st.sidebar.error("Enter a repository URL before indexing.")
        else:
            try:
                with st.status("Cloning, parsing, and indexing repository...", expanded=True) as status:
                    progress = st.progress(0, text="Preparing index operation.")
                    status.write("Loading local embedding model; the first download may take a few minutes.")
                    embedder = load_embedding_model(settings["embedding_model"])
                    status.write("Cloning repository and extracting source files.")

                    def update_progress(value: int, label: str) -> None:
                        progress.progress(value, text=label)

                    indexed = index_repository(
                        settings["url"],
                        settings["branch"],
                        settings["token"],
                        embedder,
                        chroma_client,
                        update_progress,
                    )
                    previous = registry.get(indexed.repo_id)
                    if previous:
                        indexed.active = previous.active
                    registry[indexed.repo_id] = indexed
                    save_registry(registry)
                    if previous and previous.collection_name != indexed.collection_name:
                        chroma_client.delete_collection(name=previous.collection_name)
                        old_snapshot = (APP_DIR / previous.source_dir).resolve()
                        if old_snapshot.exists():
                            remove_snapshot(old_snapshot, SOURCE_DIR)
                    progress.progress(100, text="Index complete.")
                    status.update(label="Repository indexed successfully.", state="complete")
                st.sidebar.success(
                    f"Indexed {indexed.file_count} files and {indexed.chunk_count} chunks."
                )
                st.rerun()
            except Exception as error:
                logger.exception("Repository indexing failed")
                st.sidebar.error(f"Indexing failed: {error}")

    settings["repo"] = registry.get(settings["repo"].repo_id) if settings["repo"] else None
    embedder = None
    if settings["repo"]:
        try:
            with st.status("Loading the local embedding model...", expanded=False):
                embedder = load_embedding_model(settings["embedding_model"])
        except Exception as error:
            logger.exception("Embedding model could not be loaded")
            st.sidebar.error(f"Could not load the embedding model: {error}")

    show_hero()
    chat_tab, explorer_tab, insights_tab = st.tabs(["Chat", "Explorer", "Insights"])
    with chat_tab:
        render_chat(settings["repo"], settings, chroma_client, embedder)
        if st.session_state.get("chat_history"):
            markdown = []
            for message in st.session_state["chat_history"]:
                heading = "You" if message["role"] == "user" else "RepoLens"
                markdown.append(f"## {heading}\n\n{message['content']}")
                if message.get("sources"):
                    citations = [
                        f"- `{item.chunk.path}:{item.chunk.start_line}-{item.chunk.end_line}`"
                        for item in message["sources"]
                    ]
                    markdown.append("\n".join(citations))
            st.download_button(
                "Download chat as Markdown",
                "\n\n".join(markdown),
                file_name="repolens-chat.md",
                mime="text/markdown",
            )
    with explorer_tab:
        render_explorer(settings["repo"])
    with insights_tab:
        render_insights(settings["repo"], settings)


if __name__ == "__main__":
    main()
