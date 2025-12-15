import numpy as np
from typing import List, Dict, Tuple
import json
from pathlib import Path

class VectorDatabase:
    """
    Custom vector database with flat index supporting:
    - Insert operations (single and batch)
    - Top-k search using dot product similarity
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vectors = []
        self.metadata = []
        self.ids = []
        self.next_id = 0
        
    def insert(self, vector: np.ndarray, metadata: Dict = None) -> str:
        """Insert a single vector with optional metadata"""
        if vector.shape[0] != self.dimension:
            raise ValueError(f"Vector dimension {vector.shape[0]} doesn't match database dimension {self.dimension}")
        
        doc_id = f"doc_{self.next_id}"
        self.next_id += 1
        
        self.vectors.append(vector)
        self.metadata.append(metadata or {})
        self.ids.append(doc_id)
        
        return doc_id
    
    def batch_insert(self, vectors: List[np.ndarray], metadata_list: List[Dict] = None) -> List[str]:
        """Insert multiple vectors at once"""
        if metadata_list is None:
            metadata_list = [{}] * len(vectors)
        
        if len(vectors) != len(metadata_list):
            raise ValueError("Number of vectors and metadata entries must match")
        
        doc_ids = []
        for vector, metadata in zip(vectors, metadata_list):
            doc_id = self.insert(vector, metadata)
            doc_ids.append(doc_id)
        
        return doc_ids
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Search for top-k most similar vectors using dot product similarity
        Returns: List of (doc_id, similarity_score, metadata) tuples
        """
        if len(self.vectors) == 0:
            return []
        
        if query_vector.shape[0] != self.dimension:
            raise ValueError(f"Query vector dimension {query_vector.shape[0]} doesn't match database dimension {self.dimension}")
        
        # Normalize query vector for dot product similarity
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        
        # Calculate dot product with all vectors
        similarities = []
        for i, vec in enumerate(self.vectors):
            vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
            similarity = np.dot(query_norm, vec_norm)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        k = min(k, len(similarities))
        results = []
        for i, sim in similarities[:k]:
            results.append((self.ids[i], float(sim), self.metadata[i]))
        
        return results
    
    def save(self, filepath: str):
        """Save database to disk"""
        data = {
            'dimension': self.dimension,
            'vectors': [v.tolist() for v in self.vectors],
            'metadata': self.metadata,
            'ids': self.ids,
            'next_id': self.next_id
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'VectorDatabase':
        """Load database from disk"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        db = cls(dimension=data['dimension'])
        db.vectors = [np.array(v) for v in data['vectors']]
        db.metadata = data['metadata']
        db.ids = data['ids']
        db.next_id = data['next_id']
        
        return db
    
    def __len__(self):
        return len(self.vectors)
    
    def stats(self) -> Dict:
        """Return database statistics"""
        return {
            'total_documents': len(self.vectors),
            'dimension': self.dimension,
            'next_id': self.next_id
        }