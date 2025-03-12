import streamlit as st
from pathlib import Path
import tempfile
from typing import List, Set

from src.document_processing.processor import DocumentProcessor
from src.llm.handler import LLMHandler
from src.utils.logger import get_logger

logger = get_logger()

def get_existing_collections() -> Set[str]:
    """Get all existing collections from ChromaDB"""
    doc_processor = DocumentProcessor()
    try:
        # Get all collection names from ChromaDB
        collections = doc_processor.db_client.list_collections()
        print("Collection: ", collections)
        return {collection for collection in collections}
    except Exception as e:
        logger.error(f"Error fetching collections: {str(e)}")
        return set()

def init_processors():
    """Initialize processors if they don't exist in session state"""
    if 'doc_processor' not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor()
    if 'llm_handler' not in st.session_state:
        st.session_state.llm_handler = LLMHandler()

def init_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'collections' not in st.session_state:
        # Initialize with existing collections from DB
        st.session_state.collections = get_existing_collections()

def upload_document() -> None:
    """Handle document upload and processing"""
    st.sidebar.header("Upload Document")
    
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=['pdf'])
    collection_name = st.sidebar.text_input("Collection Name", 
                                          help="Enter a name for this knowledge base")
    
    if uploaded_file and collection_name:
        if st.sidebar.button("Process Document"):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                with st.spinner('Processing document...'):
                    result = st.session_state.doc_processor.process_and_store_document(
                        file_path=tmp_path,
                        collection_name=collection_name,
                        reset_collection=True
                    )
                    
                    if "error" not in result:
                        st.session_state.collections.add(collection_name)
                        st.sidebar.success(f"Document processed: {result['chunks']} chunks created")
                    else:
                        st.sidebar.error(f"Error processing document: {result['error']}")
            finally:
                Path(tmp_path).unlink()

def display_chat_interface() -> None:
    """Display chat interface and handle interactions"""
    st.title("RAG Chat Interface")
    
    # Collection selector
    if st.session_state.collections:
        selected_collection = st.selectbox(
            "Select Knowledge Base",
            options=sorted(st.session_state.collections),
            index=0 if st.session_state.collections else None
        )
        
        # Chat interface
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your document"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Query document
                    context_results = st.session_state.doc_processor.get_chunks(
                        query=prompt,
                        collection_name=selected_collection
                    )
                    print("Context results: ", context_results)
                    
                    if not context_results:
                        response = "I couldn't find relevant information in the selected knowledge base."
                    else:
                        # Prepare context for LLM
                        context = "\n".join([res["content"] for res in context_results])
                        
                        try:
                            # Generate explanation
                            response = st.session_state.llm_handler.generate_explanation(
                                topic=prompt,
                                context=context
                            )
                            
                            # # Add sources
                            # sources = [f"Source: Page {res['metadata']['page']}" 
                            #          for res in context_results]
                            # response += "\n\n" + "\n".join(sources)
                            
                        except Exception as e:
                            logger.error(f"Error generating response: {str(e)}")
                            response = "I encountered an error generating the response."

                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("No knowledge bases available. Please upload a document to start chatting.")

def main():
    """Main function to run the Streamlit app"""
    init_session_state()
    init_processors()
    upload_document()
    display_chat_interface()

if __name__ == "__main__":
    main()