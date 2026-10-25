"""Global developer-themed Streamlit styling."""

import streamlit as st


def apply_styles() -> None:
    """Configure wide layout and a restrained dark developer theme."""
    st.set_page_config(
        page_title="RepoLens — Codebase Q&A",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        .stApp { background: #0b1020; }
        [data-testid="stHeader"] { background: rgba(11,16,32,.85); }
        [data-testid="stSidebar"] { background: #10172a; border-right: 1px solid #202a42; }
        .block-container { max-width: 1440px; padding-top: 1.25rem; padding-bottom: 3rem; }
        h1, h2, h3 { letter-spacing: -0.025em; }
        .hero { padding: 1.2rem 1.4rem; margin: 0 0 1.4rem; border: 1px solid #273354;
                border-radius: 18px; background: linear-gradient(115deg,#171e38,#11172a 70%); }
        .hero h1 { margin: 0 0 .25rem; font-size: 2rem; color: #f3f4ff; }
        .hero p { margin: 0; color: #a7b0cb; }
        .card { padding: 1rem 1.1rem; border: 1px solid #28334f; border-radius: 14px;
                background: #11182b; margin-bottom: .8rem; }
        .eyebrow { color: #a5a9ff; font-size: .77rem; font-weight: 700; text-transform: uppercase;
                   letter-spacing: .08em; }
        .citation-badge { display: inline-block; margin: .1rem .2rem; padding: .12rem .45rem;
                          border: 1px solid #5149a8; border-radius: 999px; color: #c7c2ff;
                          background: #242044; font: .78rem ui-monospace, monospace; }
        code, pre, [data-testid="stCode"] { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
        .stButton > button { border-radius: 10px; border-color: #303b5b; }
        .stButton > button[kind="primary"] { background: #6558d3; border-color: #6558d3; }
        [data-testid="stChatMessage"] { border: 1px solid #232e49; border-radius: 14px;
                                        background: rgba(18,25,44,.62); padding: .8rem; }
        [data-testid="stMetric"] { padding: .85rem; border: 1px solid #28334f; border-radius: 12px;
                                   background: #11182b; }
        .stTabs [data-baseweb="tab-list"] { gap: .4rem; }
        .stTabs [data-baseweb="tab"] { border-radius: 9px 9px 0 0; }
        @media (max-width: 800px) {
          .block-container { padding-left: 1rem; padding-right: 1rem; }
          .hero h1 { font-size: 1.6rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
