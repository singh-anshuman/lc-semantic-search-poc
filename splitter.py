from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_document():
    documents = [
        Document(
            page_content="Dogs are great companions, known for their loyalty and friendliness.",
            metadata={"source": "mammal-pets-doc"},
        ),
        Document(
            page_content="Cats are independent pets that often enjoy their own space.",
            metadata={"source": "mammal-pets-doc"},
        ),
    ]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10, chunk_overlap=5, add_start_index=True
    )
    all_splits = text_splitter.split_documents(documents)

    print(f"Total Splits: {len(all_splits)}")


if __name__ == "__main__":
    load_dotenv()
    split_document()
