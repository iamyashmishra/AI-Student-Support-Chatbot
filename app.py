import streamlit as st
from dotenv import load_dotenv
from google import genai
import os

from rag import search_documents


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()


# ==========================================
# GEMINI API KEY
# ==========================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API key not found. Please check your .env file.")
    st.stop()


# ==========================================
# GEMINI CLIENT
# ==========================================

client = genai.Client(api_key=api_key)


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="AI Student Support Chatbot",
    page_icon="🎓",
    layout="centered"
)


# ==========================================
# TITLE
# ==========================================

st.title("🎓 AI Student Support Chatbot")

st.write(
    "Ask questions about academics, examinations, "
    "scholarships, fees, hostel, and student services."
)


# ==========================================
# CHAT HISTORY
# ==========================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ==========================================
# DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# ==========================================
# USER QUESTION
# ==========================================

question = st.chat_input("Ask your question...")


if question:

    # Show user question
    with st.chat_message("user"):

        st.write(question)


    # Save question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })


    # ======================================
    # SEARCH DOCUMENTS
    # ======================================

    try:

        results = search_documents(
            question,
            number_of_results=3
        )


        # Combine retrieved information
        context = ""

        for result in results:

            context += (
                "\n\nSOURCE: "
                + result["source"]
                + "\n"
                + result["text"]
            )


        # ==================================
        # AI INSTRUCTIONS
        # ==================================

        prompt = f"""
You are an AI Student Support Assistant.

Your job is to help college students using
the information retrieved from college documents.

Rules:

1. Be polite and professional.
2. Give simple and clear answers.
3. Use the provided document information.
4. Do not invent college rules, fees, dates,
   deadlines, or policies.
5. If the retrieved documents do not contain
   the answer, clearly say that the information
   is not available in the provided documents.
6. For general academic questions, you can
   provide helpful explanations.
7. Mention the document source when useful.

RETRIEVED COLLEGE INFORMATION:

{context}

STUDENT QUESTION:

{question}
"""


        # ==================================
        # GENERATE ANSWER
        # ==================================

        with st.chat_message("assistant"):

            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )


            answer = response.text


            st.write(answer)


            # Save answer
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })


    except Exception as e:

        with st.chat_message("assistant"):

            st.error(
                f"Something went wrong: {e}"
            )