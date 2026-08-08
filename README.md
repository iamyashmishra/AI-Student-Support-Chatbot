# AI Student Support Chatbot

An intelligent student support assistant powered by **RAG (Retrieval-Augmented Generation)**. Students can ask questions about admissions, academics, exams, fees, scholarships, hostel, and placement — and get answers grounded in official college documents.

## Features

- **Document-grounded answers** — retrieves relevant sections from PDFs before responding
- **Modern chat UI** — clean Streamlit interface with quick-question shortcuts
- **Source transparency** — expandable citations showing which documents were used
- **Cached indexing** — embeddings load once per session for faster follow-up questions

## Tech Stack

| Layer | Technology |
|-------|------------|
| UI | Streamlit |
| LLM | Google Gemini |
| Embeddings | Gemini Embedding |
| Vector search | FAISS |
| PDF parsing | pypdf |

## Project Structure

```
AI_Student_Support_Chatbot/
├── app.py              # Streamlit chat interface
├── rag.py              # Document loading, embeddings, and search
├── documents/          # College PDF knowledge base
├── data/               # Supplementary text data
├── .streamlit/         # Theme and app config
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/iamyashmishra/AI-Student-Support-Chatbot.git
cd AI-Student-Support-Chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API key

```bash
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux
```

Add your Gemini API key to `.env`:

```
GEMINI_API_KEY=your_api_key_here
```

Get a key from [Google AI Studio](https://aistudio.google.com/apikey).

### 5. Run the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

## Usage

1. Open the app in your browser
2. Pick a **quick question** from the sidebar or type your own
3. Read the AI response
4. Expand **View source documents** to see which files were referenced

## Adding Documents

Place PDF files in the `documents/` folder. Restart the app to re-index them.

Supported topics include:

- College & admission brochures
- Academic regulations
- Student handbook
- Examination guidelines
- Fee structure
- Scholarship information

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API key (required) |

## Disclaimer

This chatbot answers from uploaded documents only. Always confirm fees, deadlines, and policies with official college administration.

## License

MIT
