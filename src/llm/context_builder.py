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
            # Use either standard or enhanced retriever
            if use_multi_query:
                chunks = self.retriever.retrieve_with_multi_query(
                    question=query,
                    collection_name=collection_name,
                    chunks_per_query=max_chunks,
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
            
            # Build context string
            context_parts = []
            for i, chunk in enumerate(chunks):
                content = chunk.get("content", "").strip()
                if content:
                    # Add source information if available
                    metadata = chunk.get("metdata", {})
                    source = metadata.get("sources", "unknown")

                    # Format the context with source
                    context_parts.append(f"# {content}")
        
            return "\n\n".join(context_parts)
        
        except Exception as e:
            logger.error(f"Error building context: {str(e)}")
            raise ContextError(f"Failed to build context: {str(e)}")
        
    
    def get_explanation_context(
        self, topic: str, collection_name: str,  use_multi_query: bool = True
    ) -> str:
        """Get context specifically for explanation prompts"""

        return self.build_context(
            query=topic,
            collection_name=collection_name,
            use_multi_query=True,
            max_chunks=self.max_chunks + 2
        )
            
