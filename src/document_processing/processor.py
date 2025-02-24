from typing import Dict, List
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from src.document_processing.pdf_extractor import PDFParser
from src.document_processing.utils import clean_text, validate_collection_name
from src.database.chroma_client import ChromaDBClient
from src.utils.logger import get_logger
from src.config.settings import get_settings

logger = get_logger()
settings = get_settings()


class DocumentProcessor: 
    """Handles document processing and storage"""

    def __init__(self, persist_dir: str = "db"):
        # Lzy loading for embeddings
        self._embeddings = None 
        
        self.db_client = ChromaDBClient(persist_dir=persist_dir)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                " ",
                ".",
                ",",
                "\u200b",  # Zero-width space
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic full stop
                "",
            ]
        )

        self.pdf_parser = PDFParser()

    @property
    def embeddings(self):
        """Lazy load embeddings when first needed"""
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True} 
            )
        return self._embeddings
    
    def process_and_store_document(
        self, file_path: str, collection_name: str = "collections",
        reset_collection: bool = False
    ) -> Dict: 
        """Create chunks > convert to embeddings > store in ChromaDB"""
        logger.info(f"Processing document: {file_path}")
        try:
            if not validate_collection_name(collection_name):
                raise ValueError(
                    f"Collection name must be 3-63 characters long. Cannot start or end with : numbers, underscores, hyphens"
                )
            # Reset collection 
            if reset_collection: 
                try:
                    self.db_client.delete_collection(name=collection_name)
                    logger.info(f"Deleting existing collection: {collection_name}")
                except Exception:
                    logger.warning(f"Could not delete collection: {collection_name}, moving forward ...")

            # Get or create a collection
            collection = self.db_client.get_or_create_collection(name=collection_name)

            # Parser pdf 
            raw_documents = self.pdf_parser.extract_clean_text(
                file_path=[file_path]
            )

            if not raw_documents:
                logger.error(f"No content found in document")
                return {"error": "empty document"}
            
            # Create document object
            documents = []
            for doc in raw_documents:
                documents.append(
                    Document(
                        page_content=doc.text,
                        metadata={"source": file_path}
                    )
                )

            # Split into chunks 
            chunks = self.text_splitter.split_documents(documents=documents)

            # Prepare data for ChromaDB
            texts = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                # Cleaned text content
                clean_content = clean_text(chunk.page_content)
                if not clean_content:
                    continue
                texts.append(clean_content)
                metadatas.append({
                    "source": file_path,
                    "chunk_ids": i
                })
                ids.append(f"Chunk_{i}_{Path(file_path).stem}")

            # Store in database
            if texts:
                collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )

                return {
                    "status": "success",
                    "file_path": file_path,
                    "chunks": len(chunks),
                    "collection": collection_name
                }
            else:
                return {"error": "No valid chunks created"}
            
        except Exception as e:
            logger.error(f"Document processing error: {str(e)}")
            return {"error": str(e)}

    def get_chunks(
        self, query: str, collection_name: str, n_results: int = 5
    ) -> List[Dict]:
        try:
            logger.info(f"Getting chunks for {query}")
            collection = self.db_client.get_or_create_collection(collection_name)

            # Query for chunks
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )

            # Format results
            chunks = []
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                if doc: # Skip empty chunks
                    chunks.append({
                        "content": doc, 
                        "metadata": metadata
                    })

            return chunks
        
        except Exception as e:
            logger.error(f"Error retrieving chunks: {str(e)}")
            return []


# Example usage       
if __name__ == "__main__":
    processor = DocumentProcessor()
    print(Path("data/sample.pdf"))
    print("Collections", processor.db_client.list_collections())

    stats = processor.process_and_store_document(
        file_path="data/sample.pdf",
        collection_name="test",
        reset_collection=True
    )

    print("Processing stats: ", stats)

    chunks = processor.get_chunks(
        query="How was the training data curated to train GPT-2?",
        collection_name="test",
        n_results=5
    )

    for i, chunk in enumerate(chunks):
        
        print(f"Chunk_{i}", chunk['content'])
        print(f"Metadata:", chunk['metadata'])
        


    
            


