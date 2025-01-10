from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone


def insert_pinecone_record():
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

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_1 = embeddings.embed_query(all_splits[0].page_content)
    vector_2 = embeddings.embed_query(all_splits[1].page_content)

    print(f"Generated vectors of lengths {len(vector_1)} & {len(vector_2)}\n")
    print(vector_1[:10])
    print(vector_2[:10])

    pc = Pinecone()

    index = pc.Index("lc-semantic-search-poc")
    vector_store = PineconeVectorStore(embedding=embeddings, index=index)
    ids = vector_store.add_documents(documents=all_splits)
    print(f"Record IDs: {ids}")


if __name__ == "__main__":
    load_dotenv()
    insert_pinecone_record()
