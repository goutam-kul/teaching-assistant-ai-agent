from typing import Optional, Dict, List
from src.utils.logger import get_logger
from src.config.settings import get_settings
from src.utils.exceptions import ContextError

logger = get_logger()
settings = get_settings()

class ContextBuilder:
    """Builds contexts from retrieved chunks for LLM prompts"""

    def __init__(self, enhanced_retriever):
        """Initialize with retriever"""
        self.retriever = enhanced_retriever
        self.max_chunks = settings.MAX_CONTEXT_CHUNKS

    def build_context(
        self, 
        query: str,
        collection_name: str,
        max_chunks: Optional[int] = None,
        use_multi_query: bool = True
    ) -> str:
        if max_chunks is None:
            max_chunks = self.max_chunks

        try:
            # Use either standard or enhanced retriever with ranking
            if use_multi_query:
                chunks = self.retriever.retrieve_with_multi_query(
                    question=query,
                    collection_name=collection_name,
                    chunks_per_query=max_chunks,
                    max_chunks=max_chunks,
                    deduplicate=True
                )
            else:
                chunks = self.retriever.document_processor.get_chunks(
                    query=query,
                    collection_name=collection_name,
                    n_results=max_chunks
                )
                
            if not chunks:
                logger.warning(f"No chunks found for query: {query}")
                return ""
            
            # Build context string from ranked chunks
            context_parts = []
            # Take only the top few most relevant chunks to avoid diluting the context
            for i, chunk in enumerate(chunks[:max_chunks]):
                content = chunk.get("content", "").strip()
                if content:
                    # Add source and ranking information if available
                    metadata = chunk.get("metadata", {})
                    similarity = metadata.get("similarity_score", 0)
                    source = metadata.get("source", "unknown")

                    # Add a header with ranking information for debugging 
                    context_parts.append(f"#[{i+1} Relevance: {similarity:.3f}\n{content}]")
        
            final_context = "\n\n".join(context_parts)
            logger.info(f"Build context with {len(chunks)} chunks, total_lenght: {len(final_context)} characters")
            return final_context
        
        except Exception as e:
            logger.error(f"Error building context: {str(e)}")
            raise ContextError(f"Failed to build context: {str(e)}")
        
    
    def get_explanation_context(
        self, topic: str, collection_name: str,  use_multi_query: bool = True
    ) -> str:
        """Get context specifically for explanation prompts"""

        context = self.build_context(
            query=topic,
            collection_name=collection_name,
            use_multi_query=use_multi_query,
            max_chunks=self.max_chunks 
        )

        # Log context information
        logger.info(f"Explanation context created with {len(context) if context else 0} characters")
        # print("explanation context:\n", context)  # Debug
    
        return context