from typing import List, Dict, Tuple
import json
import numpy as np
from src.utils.logger import get_logger
from src.config.settings import get_settings

logger = get_logger()
settings = get_settings()


class DocumentRanker:
    """Ranks retrieved documents based on relevancy to the original query"""

    def __init__(self, embeddings):
        self.embeddings = embeddings

    def reciprocal_rank_fusion(self, chunks_lists: List[List[Dict]], 
                               k: int = 60) -> List[Dict]:
        """
        Implements Reciprocal Rank Fusion to combine multiple ranked lists
        
        Args:
            chunks_lists: List of lists of chunks from different queries
            k: Constant to prevent division by zero and control influence of high rankings
            
        Returns:
            List of ranked chunks
        """
        fused_scores = {}
        
        # Process each list of chunks (one per query variant)
        for chunks in chunks_lists:
            for rank, chunk in enumerate(chunks):
                # Create a unique identifier for the chunk
                content = chunk.get("content", "")
                if not content:
                    continue
                    
                # Update the score using reciprocal rank fusion formula
                if content not in fused_scores:
                    fused_scores[content] = {
                        "chunk": chunk,
                        "score": 0
                    }
                
                # RRF formula: 1 / (rank + k)
                fused_scores[content]["score"] += 1 / (rank + k)
        
        # Sort by score in descending order
        sorted_chunks = [
            item["chunk"] 
            for item in sorted(
                fused_scores.values(), 
                key=lambda x: x["score"], 
                reverse=True
            )
        ]
        
        return sorted_chunks
    
    def semantic_similarity_boost(
        self, chunks: List[Dict], 
        original_query: str
    ) -> List[Dict]:
        """Boost chunk scores based on semantic similarity to original query"""
        try:
            # Get embeddings for the original query
            query_embeddings = self.embeddings.embed_query(original_query)

            # Calculate similarity for each chunk
            for chunk in chunks:
                content = chunk.get("content", "")
                if not content:
                    continue

                # Calculate cosine similarity
                chunk_embedding = self.embeddings.embed_query(content)
                similarity = np.dot(query_embeddings, chunk_embedding)

                # Add similarity score to metadata
                metadata = chunk.get("metadata", {})
                metadata["similarity_score"] = float(similarity)
                chunk["metadata"] = metadata

            # Sort by similarity score
            return sorted(chunks, key=lambda x: x.get("metadata", {}).get("similarity_score", 0), reverse=True)
        
        except Exception as e:
            logger.error(f"Error calculating similarity score for chunks: {str(e)}")
            return chunks
        
    def hybrid_rank(
        self, chunks_lists: List[List[Dict]],
        original_query: str,
        rrf_k: int = 60,
        max_chunks: int = 5
    ) -> List[Dict]:
        """Combine reciprocal rank fusion and semantic similarity"""
        # First apply reciprocal rank fusion
        fused_chunks = self.reciprocal_rank_fusion(chunks_lists=chunks_lists, k=rrf_k)

        # Then boost with semantic similarity
        ranked_chunks = self.semantic_similarity_boost(chunks=fused_chunks, original_query=original_query)

        # Log ranking information
        logger.info(f"Ranked {len(ranked_chunks)} chunks by relevance to : '{original_query}'")

        # Return top chunks - fixed to return ranked_chunks instead of chunks_lists
        return ranked_chunks[:max_chunks]