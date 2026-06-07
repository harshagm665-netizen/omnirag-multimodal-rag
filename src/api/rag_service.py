from typing import Dict, Any, List
from groq import Groq
from src.embeddings.colpali_embedder import ColpaliEmbedder
from src.database.qdrant_client import QdrantStore
from src.llm.key_rotator import KeyRotator
import logging

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, embedder: ColpaliEmbedder, qdrant: QdrantStore, key_rotator: KeyRotator):
        self.embedder = embedder
        self.qdrant = qdrant
        self.key_rotator = key_rotator

    def _get_groq_client(self) -> Groq:
        """Fetch a new Groq client using rotated API key."""
        api_key = self.key_rotator.get_next_key()
        return Groq(api_key=api_key)

    def query(self, user_query: str) -> Dict[str, Any]:
        """
        Executes a RAG query:
        1. Embed the user query.
        2. Retrieve top-k relevant pages from Qdrant.
        3. Pass query + retrieved context to LLM with citation instructions.
        4. Return response + citation metadata.
        """
        # Step 1: Embed Query
        logger.info("Embedding user query...")
        query_vector = self.embedder.embed_query(user_query)

        # Step 2: Retrieve from Vector Store
        logger.info("Retrieving contexts from Qdrant...")
        retrieved_contexts = self.qdrant.search_similar(query_vector, top_k=3)

        if not retrieved_contexts:
            return {
                "answer": "I'm sorry, I couldn't find any relevant documents to answer your query.",
                "citations": []
            }

        # Format Context
        context_text = ""
        citations = []
        for idx, item in enumerate(retrieved_contexts):
            # In a full multimodal system, we would pass the actual image patch back to a Vision-Language LLM.
            # For this MVP, we simulate passing the retrieved image metadata and textual representation if available.
            source = item.get("source", "Unknown")
            page = item.get("page_number", "Unknown")
            text_content = item.get("extracted_text", f"[Visual content from {source}, Page {page}]")
            
            context_text += f"\n--- Document {idx+1} ---\nSource: {source}\nPage: {page}\nContent: {text_content}\n"
            citations.append({"source": source, "page": page, "score": item.get("_score")})

        # Step 3: LLM Generation
        prompt = f"""You are an enterprise AI assistant. Answer the user's query based ONLY on the provided context below.
If the context does not contain the answer, say "I don't know based on the provided documents."

For every fact you state, you MUST append an inline citation in the exact format: [Source: X, Page: Y]

Context:
{context_text}

User Query: {user_query}
"""
        logger.info("Generating response with Groq (rotating key)...")
        client = self._get_groq_client()
        
        try:
            # We use Llama-3.3-70b-versatile on Groq for ultra-fast generation
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a highly precise enterprise AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=1024
            )
            answer = completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API Error: {e}")
            answer = f"Error during generation: {e}"

        return {
            "answer": answer,
            "citations": citations
        }
