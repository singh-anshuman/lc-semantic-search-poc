from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader


def semantic_search():
    file_path = "./data/flight_ticket.pdf"
    loader = PyPDFLoader(file_path)

    docs = loader.load()

    print(f"Document Length: {len(docs)}")
    print(f"{docs[1].page_content[:200]}\n")
    print(docs[1].metadata)


if __name__ == "__main__":
    load_dotenv()
    semantic_search()
