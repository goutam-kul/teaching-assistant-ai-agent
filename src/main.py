from src.document_processing.processor import DocumentProcessor
from src.retrieval.enhanced_retriever import EnhancedRetriever
from src.llm.context_builder import ContextBuilder
from src.llm.handler import LLMHandler
from src.utils.logger import get_logger

logger = get_logger()

def initialize_components():
    """Initialize all components for the application"""
    # Initialize document processor
    processor = DocumentProcessor(persist_dir="db")
    
    # Initialize enhanced retriever
    retriever = EnhancedRetriever(document_processor=processor)
    
    # Initialize context builder
    context_builder = ContextBuilder(enhanced_retriever=retriever)
    
    # Initialize LLM handler
    llm_handler = LLMHandler(context_builder=context_builder)
    
    return {
        "processor": processor,
        "retriever": retriever,
        "context_builder": context_builder,
        "llm_handler": llm_handler
    }

def process_document(processor, file_path, collection_name, reset=False):
    """Process a document and store in the database"""
    result = processor.process_and_store_document(
        file_path=file_path,
        collection_name=collection_name,
        reset_collection=reset
    )
    return result

def answer_question(llm_handler, question, collection_name, use_multi_query=True):
    """Answer a question using the LLM with enhanced retrieval"""
    answer = llm_handler.explain_topic(
        topic=question,
        collection_name=collection_name,
        use_multi_query=use_multi_query
    )
    return answer

def main():
    # Initialize components
    components = initialize_components()
    processor = components["processor"]
    llm_handler = components["llm_handler"]
    
    # Example usage
    file_path = "data/sample.pdf"
    collection_name = "ncert"
    
    # Process document (uncomment if needed)
    # result = process_document(
    #     processor=processor,
    #     file_path=file_path,
    #     collection_name=collection_name,
    #     reset=False
    # )
    # print(f"Document processing result: {result}")
    
    # Answer questions
    while True:
        question = input("\nEnter your question (or 'q' to quit): ")
        if question.lower() == 'q':
            break
            
        print("\nGenerating answer...")
        answer = answer_question(
            llm_handler=llm_handler,
            question=question,
            collection_name=collection_name,
            use_multi_query=True  # Set to False to disable multi-query
        )
        
        print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()