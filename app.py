import os

import streamlit as st
from dotenv import load_dotenv
from google import genai

from rag import get_rag_index

st.set_page_config(
    page_title="Student Support Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

load_dotenv()

SUGGESTED_QUESTIONS = [
    "What scholarships are available?",
    "What is the fee structure?",
    "How do exam guidelines work?",
    "What are the hostel rules?",
    "Where is placement support?",
]

SYSTEM_PROMPT = """You are a friendly and professional AI Student Support Assistant.

Help college students using ONLY the retrieved college document information.

Rules:
1. Be polite, concise, and easy to understand.
2. Use bullet points or short paragraphs when helpful.
3. Do not invent fees, dates, deadlines, rules, or policies.
4. If the documents do not contain the answer, say clearly that the information is not available in the provided documents and suggest contacting the relevant college office.
5. Mention the document source when it helps the student verify details.

RETRIEVED COLLEGE INFORMATION:
{context}

STUDENT QUESTION:
{question}
"""


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            /* Main content — leave room for fixed chat input */
            section.main > div.block-container {
                padding-top: 1rem;
                padding-bottom: 7rem;
                max-width: 900px;
            }

            /* Sidebar */
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #1E1B4B 0%, #312E81 100%);
            }

            [data-testid="stSidebar"] * {
                color: #F8FAFC !important;
            }

            [data-testid="stSidebar"] .stButton > button {
                background: rgba(255, 255, 255, 0.12);
                border: 1px solid rgba(255, 255, 255, 0.18);
                color: #F8FAFC !important;
                border-radius: 10px;
                text-align: left;
                width: 100%;
            }

            [data-testid="stSidebar"] .stButton > button:hover {
                background: rgba(255, 255, 255, 0.22);
            }

            /* Compact header */
            .app-header {
                background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
                border-radius: 14px;
                padding: 1rem 1.25rem;
                color: white;
                margin-bottom: 0.75rem;
            }

            .app-header h2 {
                margin: 0;
                font-size: 1.35rem;
                color: white !important;
            }

            .app-header p {
                margin: 0.25rem 0 0 0;
                font-size: 0.9rem;
                opacity: 0.92;
            }

            /* Welcome hint */
            .welcome-hint {
                background: #EEF2FF;
                border: 1px solid #C7D2FE;
                border-radius: 12px;
                padding: 0.85rem 1rem;
                color: #3730A3;
                font-size: 0.92rem;
                margin-bottom: 0.75rem;
                text-align: center;
            }

            /* Chat messages */
            div[data-testid="stChatMessage"] {
                border-radius: 12px;
            }

            /* ── FIX: Make chat input always visible at bottom ── */
            [data-testid="stBottomBlockContainer"] {
                position: fixed !important;
                bottom: 0 !important;
                left: 0 !important;
                right: 0 !important;
                z-index: 9999 !important;
                background: #FFFFFF !important;
                border-top: 2px solid #4F46E5 !important;
                box-shadow: 0 -6px 24px rgba(79, 70, 229, 0.12) !important;
                padding: 0.75rem 1rem 1rem 1rem !important;
            }

            [data-testid="stChatInput"] {
                max-width: 900px;
                margin: 0 auto;
            }

            [data-testid="stChatInput"] textarea {
                border: 2px solid #C7D2FE !important;
                border-radius: 14px !important;
                padding: 0.75rem 1rem !important;
                font-size: 1rem !important;
                background: #F8FAFC !important;
                color: #0F172A !important;
                min-height: 52px !important;
            }

            [data-testid="stChatInput"] textarea:focus {
                border-color: #4F46E5 !important;
                box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15) !important;
            }

            /* Hint label above input area */
            [data-testid="stBottomBlockContainer"]::before {
                content: "Type your question below ↓";
                display: block;
                text-align: center;
                font-size: 0.78rem;
                font-weight: 600;
                color: #4F46E5;
                margin-bottom: 0.4rem;
                letter-spacing: 0.02em;
            }

            /* Scrollable chat area styling */
            [data-testid="stVerticalBlockBorderWrapper"] {
                border-radius: 12px;
            }

            /* Hide Streamlit footer to avoid overlap */
            footer { visibility: hidden; }
            footer::after {
                content: "";
                visibility: visible;
                display: block;
                height: 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading college documents and building search index...")
def load_rag_index():
    return get_rag_index()


def render_sidebar() -> int:
    with st.sidebar:
        st.markdown("### 🎓 Student Support")
        st.caption("Ask about admissions, exams, fees, scholarships & more.")

        st.markdown("---")
        st.markdown("**Quick questions**")

        for question in SUGGESTED_QUESTIONS:
            if st.button(question, key=f"suggest_{question}"):
                st.session_state.pending_question = question
                st.rerun()

        st.markdown("---")
        num_results = st.slider("Sources to retrieve", 1, 5, 3)

        if st.button("🗑 Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_question = None
            st.session_state.last_sources = None
            st.session_state.last_error = None
            st.rerun()

        st.markdown("---")
        st.markdown("**📚 Knowledge base**")
        st.markdown(
            "- College & admission brochures\n"
            "- Academic regulations\n"
            "- Student handbook\n"
            "- Exams, fees & scholarships"
        )

        st.markdown("---")
        st.caption(
            "Answers come from uploaded documents. "
            "Verify important details with college offices."
        )

    return num_results


def render_header(doc_count: int, chunk_count: int, has_messages: bool) -> None:
    if has_messages:
        st.markdown(
            """
            <div class="app-header">
                <h2>🎓 Student Support Chat</h2>
                <p>Ask follow-up questions anytime using the box at the bottom.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="app-header">
                <h2>🎓 AI Student Support Chatbot</h2>
                <p>Instant answers from {doc_count}+ official college documents
                   ({chunk_count} searchable sections)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def format_context(results: list[dict]) -> str:
    return "\n\n".join(
        f"SOURCE: {result['source']}\n{result['text']}" for result in results
    )


def generate_answer(client: genai.Client, question: str, context: str) -> str:
    prompt = SYSTEM_PROMPT.format(context=context, question=question)
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )
    return response.text


def handle_question(
    question: str,
    client: genai.Client,
    rag_index,
    num_results: int,
) -> None:
    st.session_state.messages.append({"role": "user", "content": question})

    try:
        results = rag_index.search(question, number_of_results=num_results)
        context = format_context(results)
        answer = generate_answer(client, question, context)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.last_sources = results
    except Exception as exc:
        error_message = "Sorry, something went wrong. Please try again."
        st.session_state.messages.append({"role": "assistant", "content": error_message})
        st.session_state.last_error = str(exc)


def main() -> None:
    inject_styles()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Gemini API key not found. Copy `.env.example` to `.env` and add your key.")
        st.stop()

    client = genai.Client(api_key=api_key)

    try:
        rag_index = load_rag_index()
    except Exception as exc:
        st.error("Could not load the knowledge base.")
        st.exception(exc)
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = None
    if "last_error" not in st.session_state:
        st.session_state.last_error = None

    num_results = render_sidebar()

    has_messages = bool(st.session_state.messages)
    doc_count = len({chunk["source"] for chunk in rag_index.chunks})
    render_header(doc_count, len(rag_index.chunks), has_messages)

    # Quick-pick chips only when chat is empty (compact, single row)
    if not has_messages:
        st.markdown(
            '<div class="welcome-hint">'
            "👋 Welcome! Type your question in the <strong>box at the bottom</strong> "
            "or tap a quick question below."
            "</div>",
            unsafe_allow_html=True,
        )
        chip_cols = st.columns(len(SUGGESTED_QUESTIONS))
        for col, q in zip(chip_cols, SUGGESTED_QUESTIONS):
            with col:
                if st.button(q, key=f"chip_{q}", use_container_width=True):
                    st.session_state.pending_question = q
                    st.rerun()

    # Scrollable chat history — keeps input visible at bottom
    chat_height = 420 if not has_messages else 520
    chat_box = st.container(height=chat_height, border=True)

    with chat_box:
        if not st.session_state.messages:
            st.markdown(
                "_Your conversation will appear here. "
                "Use the prompt box fixed at the bottom of the screen to ask a question._"
            )
        else:
            for i, message in enumerate(st.session_state.messages):
                avatar = "🧑‍🎓" if message["role"] == "user" else "🤖"
                with st.chat_message(message["role"], avatar=avatar):
                    st.markdown(message["content"])

            if st.session_state.last_sources:
                with st.expander("📄 View source documents from last answer"):
                    for i, result in enumerate(st.session_state.last_sources, start=1):
                        st.markdown(f"**{i}. {result['source']}**")
                        preview = result["text"][:500]
                        st.caption(preview + ("..." if len(result["text"]) > 500 else ""))

            if st.session_state.last_error:
                st.caption(f"Last error: {st.session_state.last_error}")

    # Chat input — must stay near the end of the script
    question = st.chat_input(
        "Ask about academics, fees, exams, scholarships, hostel...",
        key="main_chat_input",
    )

    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

    if question:
        with st.status("Searching documents and generating answer...", expanded=True):
            handle_question(question, client, rag_index, num_results)
        st.rerun()


if __name__ == "__main__":
    main()
