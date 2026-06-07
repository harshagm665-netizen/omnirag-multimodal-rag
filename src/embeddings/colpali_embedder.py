import os
try:
    import torch
except ImportError:
    torch = None
from typing import List, Dict, Any
from src.ingestion.document_parser import DocumentPage

# Note: In production, install colpali-engine or use transformers.
# pip install colpali-engine

class ColpaliEmbedder:
    def __init__(self, model_name: str = "vidore/colpali-v1.2", device: str = None, mock: bool = False):
        self.mock = mock
        if device is None:
            self.device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device
            
        if not self.mock:
            if torch is None:
                raise ImportError("torch is required when mock=False")
            import logging
            logging.info(f"Loading ColPali model {model_name} on {self.device}...")
            # We use transformers/colpali integration
            from transformers import AutoProcessor, AutoModel
            
            # Using bfloat16 for efficiency if on CUDA
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(self.device).eval()
        else:
            self.dim = 128 # Fake dimension for mock

    def embed_pages(self, pages: List[DocumentPage]) -> List[Dict[str, Any]]:
        """
        Embeds a list of DocumentPages into ColPali multi-vector representations.
        Returns a list of dicts containing the vectors and the metadata.
        """
        results = []
        if not pages:
            return results

        if self.mock:
            # Generate random mock embeddings using standard library random
            import random
            for page in pages:
                results.append({
                    "vector": [random.random() for _ in range(self.dim)],
                    "metadata": page.metadata
                })
            return results

        images = [page.image for page in pages]
        
        # Preprocess
        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            
            # ColPali uses pooled embeddings or multi-vector embeddings depending on setup.
            # We'll assume the pooled output for simplicity in Qdrant indexing, or 
            # if using true ColPali multi-vector, we take the embeddings and flatten/pool them.
            # A standard approach for dense retrieval with ColPali is to take the mean of the patch embeddings.
            embeddings = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state.mean(dim=1)
            
            embeddings = embeddings.cpu().float().numpy()

        for i, page in enumerate(pages):
            results.append({
                "vector": embeddings[i].tolist(),
                "metadata": page.metadata
            })
            
        return results

    def embed_query(self, query: str) -> List[float]:
        if self.mock:
            import random
            return [random.random() for _ in range(self.dim)]
            
        with torch.no_grad():
            inputs = self.processor(text=query, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            # Take mean of token embeddings
            embedding = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state.mean(dim=1)
            return embedding[0].cpu().float().numpy().tolist()
