"""Discover and load the IronStore PDF documents with department/category metadata."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

DOCS_ROOT = Path(__file__).resolve().parents[2] / "documents" / "documents"
INTERNAL_DOCS_DIR = DOCS_ROOT / "internal_docs_by_area"
PRODUCT_CATALOG_DIR = DOCS_ROOT / "product_catalog"


def load_internal_docs():
  """Load department policy PDFs, tagging each with its department folder name."""
  documents = []
  for department_dir in sorted(INTERNAL_DOCS_DIR.iterdir()):
    if not department_dir.is_dir():
      continue
    for pdf_path in sorted(department_dir.glob("*.pdf")):
      pages = PyPDFLoader(str(pdf_path)).load()
      for page in pages:
        page.metadata.update({
          "source_type": "internal_policy",
          "department": department_dir.name,
          "file_name": pdf_path.name,
        })
      documents.extend(pages)
  return documents


def load_product_catalog():
  """Load product catalog PDFs, deriving the category from the file name."""
  documents = []
  for pdf_path in sorted(PRODUCT_CATALOG_DIR.glob("*.pdf")):
    category = (
      pdf_path.stem.replace("IronStore_Product_Catalog_", "")
      .replace("IronStore_Old_Product_Catalog_2021", "Old Catalog (2021, outdated)")
      .replace("_", " ")
    )
    pages = PyPDFLoader(str(pdf_path)).load()
    for page in pages:
      page.metadata.update({
        "source_type": "product_catalog",
        "category": category,
        "file_name": pdf_path.name,
      })
    documents.extend(pages)
  return documents


def load_all_documents():
  """Load every document in the corpus, internal policies and product catalog alike."""
  return load_internal_docs() + load_product_catalog()


if __name__ == "__main__":
  docs = load_all_documents()
  print(f"Loaded {len(docs)} pages total.")
