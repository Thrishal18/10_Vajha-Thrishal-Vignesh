from typing import List, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document


class ChromaUtils:
    def __init__(
        self,
        embedding_function,
        collection_name: str,
        chroma_api_key: str,
        tenant: str,
        database: str,
    ):
        """
        Initialize Chroma Cloud connection
        """
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            chroma_cloud_api_key=chroma_api_key,
            tenant=tenant,
            database=database,
        )

    def retrieve_context(
        self,
        query: str,
        k: int = 3,
        category: Optional[str] = None,
    ) -> str:
        """
        Retrieve relevant context from ChromaDB Cloud and return as a single string
        """

        search_kwargs = {"k": k}

        if category:
            search_kwargs["filter"] = {"category": category}

        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                **search_kwargs,
                "fetch_k": k,
                "lambda_mult": 0.7,
            },
        )

        docs: List[Document] = retriever.get_relevant_documents(query)

        if not docs:
            return ""

        # Merge retrieved chunks into one context string
        context = "\n\n".join(doc.page_content for doc in docs)

        return context
