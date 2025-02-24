from typing import List, Dict, Optional
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import  StrOutputParser
from langchain_ollama import ChatOllama
from src.config.settings import get_settings
from src.utils.logger import get_logger
from src.llm.prompts import TEMPLATE

settings = get_settings()
logger = get_logger()

class EnhancedRetriever:
    """Enhanced retrieval using multi-query generation"""

    def __init__(self, document_processor):
        """Initialize with document processor"""
        self.document_processor = document_processor
        self.llm = ChatOllama(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            base_url=settings.OLLAMA_HOST
        )

        self.query_template = TEMPLATE

        self.prompt = ChatPromptTemplate([self.query_template])

        # Create pipeline for generating queries
        self.generate_queries = (
            self.prompt
            | self.llm
            | StrOutputParser()
            | (lambda x:  [q.strip() for q in x.split("\n") if q.strip()])
        )

    def generate_query_variants(self, question: str) -> List[str]:
        """Generate different version of the questions"""
        try:
            logger.info(f"Generating variants for the original question")
            variants = self.generate_queries.invoke(question)
            # Add the original question if not already present
            if question not in variants:
                variants.insert(0, question)

            # logger.info(f"Generated query variants: {variants}")

            return variants
        except Exception as e:
            logger.error(f"Error generating query variants: {str(e)}")
            # Fallback to just original question
            return [question]
        
    def retrieve_with_multi_query(
        self,
        question: str,
        collection_name: str,
        chunks_per_query: int = 3,
        deduplicate: bool = True
    ) -> List[Dict]:
        """Retrieve chunks using multiquery variants"""
        # Generate query variants
        # Generate query variants
        query_variants = self.generate_query_variants(question)
        
        # Collect chunks from all queries
        all_chunks = []
        seen_chunks = set()
        
        for query in query_variants:
            chunks = self.document_processor.get_chunks(
                query=query,
                collection_name=collection_name,
                n_results=chunks_per_query
            )
            
            for chunk in chunks:
                # Create a unique identifier for deduplication
                if deduplicate:
                    chunk_content = chunk.get("content", "")
                    if chunk_content and chunk_content in seen_chunks:
                        continue
                    seen_chunks.add(chunk_content)
                
                # Add query information to metadata
                chunk_metadata = chunk.get("metadata", {})
                chunk_metadata["query"] = query
                chunk["metadata"] = chunk_metadata
                
                all_chunks.append(chunk)
        
        return all_chunks