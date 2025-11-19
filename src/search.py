import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from openai import OpenAI

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 llm_model: str = "gpt-4.1"):
        
        # -----------------------------
        # Load / Build FAISS Vectorstore
        # -----------------------------
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)

        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        # -----------------------------
        # Initialize OpenAI Client
        # -----------------------------
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.llm_model = llm_model
        print(f"[INFO] OpenAI LLM initialized: {llm_model}")

    # ----------------------------------------
    # RAG Search + LLM Summary
    # ----------------------------------------
    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]

        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."

        prompt = f"""
Summarize the following context for the query: '{query}'.

Context:
{context}

Summary:
"""

        # ---- Call OpenAI Chat Completion ----
        response = self.client.responses.create(
            model=self.llm_model,
            input=[{"role": "user", "content": prompt}]
        )

        # Extract text
        summary = response.output_text
        return summary


# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is YuNgong Zhang's email?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
