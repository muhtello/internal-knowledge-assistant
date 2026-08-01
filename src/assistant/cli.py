"""Interactive terminal chat loop for the IronStore assistant.

Run with:
    python -m src.assistant.cli
"""

from src.assistant.chain import build_chain


def main():
  print("IronStore Assistant — ask a question about company policy or products.")
  print("Type 'exit' or 'quit' to stop.\n")

  chain, retriever = build_chain()

  while True:
    question = input("You: ").strip()
    if not question:
      continue
    if question.lower() in {"exit", "quit"}:
      break

    answer = chain.invoke(question)
    print(f"\nAssistant: {answer}\n")

    sources = retriever.invoke(question)
    seen = set()
    for doc in sources:
      file_name = doc.metadata.get("file_name", "unknown file")
      if file_name not in seen:
        seen.add(file_name)
    if seen:
      print(f"(Sources consulted: {', '.join(sorted(seen))})\n")


if __name__ == "__main__":
  main()
