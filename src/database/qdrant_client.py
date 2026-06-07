import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

class QdrantStore:
    def __init__(self, collection_name: str = "multimodal_rag", vector_size: int = 128, in_memory: bool = True):
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        if in_memory:
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(path="./qdrant_data")
            
        self._ensure_collection()

    def _ensure_collection(self):
        # Check if collection exists
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def upsert_vectors(self, data: List[Dict[str, Any]]):
        """
        Inserts a list of dicts.
        Each dict should have: 'vector' (List[float]) and 'metadata' (Dict).
        """
        if not data:
            return

        points = []
        for item in data:
            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=item["vector"],
                    payload=item.get("metadata", {})
                )
            )
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search_similar(self, query_vector: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search the collection for similar vectors.
        Returns the payload metadata for the top_k results.
        """
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k
        )
        
        # Return payloads enriched with the score
        retrieved = []
        for hit in results.points:
            item = hit.payload.copy()
            item["_score"] = hit.score
            retrieved.append(item)
            
        return retrieved
