import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from pathlib import Path
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY is not set")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
vector_store = InMemoryVectorStore(embedding=embeddings)


def load_pdf(file_path: str) -> list[Document]:
    reader = pypdf.PdfReader(file_path)

    return [
        Document(
            page_content=text,
            metadata={"source": file_path, "page": page_number},
        )
        for page_number, page in enumerate(reader.pages)
        if (text := (page.extract_text() or "").strip())
    ]

file_path = Path("./data/nke-10k-2023.pdf")

docs = load_pdf(str(file_path.absolute()))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

all_splits = text_splitter.split_documents(docs)

ids = vector_store.add_documents(all_splits)

# @chain
# def retriever(query: str) -> List[Document]:
#     return vector_store.similarity_search(query, k=1)


# retriever.batch(
#     [
#         "How many distribution centers does Nike have in the US?",
#         "When was Nike incorporated?",
#     ],
# )

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

results = retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)

for result in results:
    print(result)
print(retriever.invoke("How many distribution centers does Nike have in the US?"))