from typing import Dict, List
from dotenv import load_dotenv
from llama_cloud_services import LlamaParse
from llama_index.core import SimpleDirectoryReader
from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger()
settings = get_settings()
load_dotenv(override=True)

class PDFParser: 
    """Simple PDF parser that extracts clean text from documents"""

    def __init__(self):
        self.parser = LlamaParse(
            result_type="markdown",   # "markdown" or "text" 
        )
        self.file_extractor = {".pdf": self.parser}

    
    def extract_clean_text(self, file_path: List):
        """Extract text from pdf"""
        logger.info(f"Extracting text from PDF: {file_path}")
        try:
            # Parse documents
            documents = SimpleDirectoryReader(
                input_files=file_path, file_extractor=self.file_extractor
            ).load_data()

            if not documents:
                logger.warning("The PDF seems to be empty")
                return {
                    "status": "error",
                    "message": "No content found"
                }
            
            return documents
        
        except Exception as e:
            logger.error(f"Failed to parse PDF: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to parse PDF: {str(e)}"
            }
    

# Example usage
if __name__ == "__main__":
    parser = PDFParser()
    docs = parser.extract_clean_text(file_path=["data/sample.pdf"])
    print(docs[0].text[:100])



