from typing import List, Dict, Optional
import re
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from src.config.settings import get_settings
from src.utils.logger import get_logger
from src.llm.prompts import TEMPLATE
from src.document_processing.ranking import DocumentRanker

settings = get_settings()
logger = get_logger()

class EnhancedRetriever:
    """Enhanced retrieval using multi-query generation"""

    def __init__(self, document_processor):
        """Initialize with document processor"""
        self.document_processor = document_processor
        self.ranker = DocumentRanker(embeddings=document_processor.embeddings)
        self.llm = ChatOllama(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            base_url=settings.OLLAMA_HOST
        )

        self.query_template = TEMPLATE

        self.prompt = ChatPromptTemplate.from_template(self.query_template)

        # Create pipeline for generating queries
        self.generate_queries = (
            self.prompt
            | self.llm
            | StrOutputParser()
            | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
        )

    def generate_query_variants(self, question: str) -> List[str]:
        """Generate different versions of the question"""
        try:
            logger.info(f"Generating variants for the original question")
            variants = self.generate_queries.invoke({"question": question})
            
            # Clean up the variants
            clean_variants = []
            for variant in variants:
                # Remove numeric prefixes like "1. ", "2: ", etc.
                clean_variant = re.sub(r'^\d+[\.\:]?\s*', '', variant).strip()
                # Remove any "endif" or similar artifacts
                clean_variant = re.sub(r'\s*endif$', '', clean_variant).strip()
                
                if clean_variant and len(clean_variant) > 5:  # Only keep substantial variants
                    clean_variants.append(clean_variant)
            
            # Add the original question if not already present
            if question not in clean_variants:
                clean_variants.insert(0, question)
                
            # Limit to a reasonable number of variants
            clean_variants = clean_variants[:3]  # Limiting to 3 variants for focus
            
            logger.info(f"Generated {len(clean_variants)} query variants")
            return clean_variants
        except Exception as e:
            logger.error(f"Error generating query variants: {str(e)}")
            # Fallback to just original question
            return [question]
        
    def retrieve_with_multi_query(
        self,
        question: str,
        collection_name: str,
        chunks_per_query: int = 3,
        max_chunks: int = 5,
        deduplicate: bool = True
    ) -> List[Dict]:
        """Retrieve chunks using multiquery variants"""
        # Generate query variants
        query_variants = self.generate_query_variants(question)
        
        # Collect chunks from all queries
        all_chunks_lists = []

        # Keep track of each query's results separately for ranking
        for query in query_variants:
            chunks = self.document_processor.get_chunks(
                query=query,
                collection_name=collection_name,
                n_results=chunks_per_query
            )

            # Add more information to metadata
            for chunk in chunks:
                chunk_metadata = chunk.get("metadata", {})
                chunk_metadata["query"] = query
                chunk["metadata"] = chunk_metadata

            # Store this query's results as a separate list
            all_chunks_lists.append(chunks)

        # Use the ranker to rank both reciprocal rank fusion and similarity
        ranked_chunks = self.ranker.hybrid_rank(
            chunks_lists=all_chunks_lists,
            original_query=question,
            max_chunks=max_chunks
        )

        # Log information about retrieved chunks
        logger.info(f"Retrieved {len(ranked_chunks)} ranked chunks for query: {question}")

        return ranked_chunks