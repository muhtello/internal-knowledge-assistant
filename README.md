# Internal Knowledge Assistant

RAG-based assistant that answers employee questions from internal company documents (HR, Finance, IT, Legal, Sales, Deliveries, Customer Support, product catalog).

## Stack
LangChain + OpenAI (gpt-4o-mini, text-embedding-3-small) + Chroma vector store.

## Setup
```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt
copy .env.example .env   # add your OPENAI_API_KEY
```

## Usage
```bash
venv\Scripts\python -m src.ingest.build_index   # build the index (run once, or after doc changes)
venv\Scripts\python -m src.assistant.cli        # chat with the assistant
```

## Notes
- `documents/` isn't tracked in git — add your own PDFs before ingesting.
- Answers are grounded only in retrieved document context; the assistant says so when it can't find an answer.
