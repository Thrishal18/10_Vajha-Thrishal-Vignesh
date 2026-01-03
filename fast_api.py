import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from chroma_utils import ChromaUtils

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

chroma = ChromaUtils(
    embedding_function=embeddings,
    collection_name="my_cloud_collection",
    chroma_api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
)

context = chroma.retrieve_context(
    query="I haven't received my refund",
    k=3,
    category="Billing"
)

print(context)
