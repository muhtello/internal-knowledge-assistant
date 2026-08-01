"""Build the retrieval-augmented generation (RAG) chain for the IronStore assistant."""

from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

PERSIST_DIR = Path(__file__).resolve().parents[2] / "chroma_db"
COLLECTION_NAME = "ironstore_docs"

SYSTEM_PROMPT = """You are IronStore's internal assistant, helping employees find \
information from official company documents (HR, Finance, IT, Legal, Sales, \
Deliveries, Customer Support, and the product catalog).

Rules:
- Answer ONLY using the provided context. Do not use outside knowledge.
- If the context does not contain enough information to answer confidently, say so \
plainly (e.g. "I don't have enough information in the company documents to answer \
that.") instead of guessing.
- If the context includes conflicting or outdated information (e.g. an old vs. \
current product catalog, or two policies that disagree), point out the conflict \
and prefer the source that looks most current rather than picking silently.
- Always cite the source documents you used, using the department/category and \
file name shown in the context.
- Be concise and direct.

Context:
{context}"""


def format_docs(docs):
  """Render retrieved chunks with their source metadata so the model can cite them."""
  blocks = []
  for doc in docs:
    label = doc.metadata.get("department") or doc.metadata.get("category", "Unknown")
    file_name = doc.metadata.get("file_name", "unknown file")
    blocks.append(f"[Source: {label} — {file_name}]\n{doc.page_content}")
  return "\n\n---\n\n".join(blocks)


def build_chain():
  """Assemble the retriever + prompt + LLM pipeline as a single runnable chain."""
  embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
  vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(PERSIST_DIR),
  )
  retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

  prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
  ])
  llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

  chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
  )
  # Expose the retriever too, so the CLI can show which files were used.
  return chain, retriever
