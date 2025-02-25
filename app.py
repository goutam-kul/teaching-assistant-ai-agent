import streamlit as st
import tempfile
from pathlib import Path
from typing import List, Set
from src.document_processing.processor import DocumentProcessor
from src.llm.handler import LLMHandler
from src.retrieval.enhanced_retriever import EnhancedRetriever
from src.llm.context_builder import ContextBuilder
from src.utils.logger import get_logger

logger = get_logger()

def get_existing_collection() -> Set[str]:
    """Get all existing collections from ChromaDB"""
    doc_processor = DocumentProcessor(persist_dir="db")
    try:
        # Get existing collection 
        collections = doc_processor.db_client.list_collections()
        logger.info(f"Existing Collection: {collections}")
        return {coll_name for coll_name in collections}
    
    except Exception as e:
        logger.error(f"Error fetching collections: {str(e)}")
        return set()
    
def init_processors():
    """Initialize processors"""
    if 'doc_processor' not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor()
    if 'retriever' not in st.session_state:
        st.session_state.retriever = EnhancedRetriever(
            document_processor=st.session_state.doc_processor
        )
    if 'context_builder' not in st.session_state:
        st.session_state.context_builder = ContextBuilder(
            enhanced_retriever=st.session_state.retriever
        )
    if 'llm_handler' not in st.session_state:
        st.session_state.llm_handler = LLMHandler(
            context_builder=st.session_state.context_builder
        )


def init_session_state():
    """Initialize session state variable"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'collections' not in st.session_state:
        # Get collection from defined method
        st.session_state.collections = get_existing_collection()


def upload_document():
    """Hanlder document upload and processing"""
    st.sidebar.header("Upload Document")

    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])
    collection_name = st.sidebar._text_input(
        "Collection Name", help="Enter a collection name for this knowlede base"
    )

    if uploaded_file and collection_name:
        if st.sidebar.button("Create Knowledge Base"):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                with st.spinner("Building Knowledge Base..."):
                    result = st.session_state.doc_processor.process_and_store_document(
                        file_path=tmp_path,
                        collection_name=collection_name,
                        reset_collection=True
                    )

                    if "error" not in result:
                        st.session_state.collections.add(collection_name)
                        st.sidebar.success(f"Document added to knowledge base")
                    else:
                        st.sidebar.error(f"Error processing document: {result['error']}")
            finally:
                Path(tmp_file).unlink()

def display_chat_interface() -> None:
    """Display chat interface and handle interactions"""
    st.title("Chat Interface")

    # Collection selector
    if st.session_state.collections:
        selected_collection = st.selectbox(
            label="Select Knowledge Base",
            options=sorted(st.session_state.collections),
            index=0 if st.session_state.collections else None
        )
    else:
        st.warning("No knowledge base available. Please create on first.")

    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input 
    if prompt := st.chat_input("Ask a question from knowledge base"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Query results
                    response = st.session_state.llm_handler.explain_topic(
                        topic=prompt,
                        collection_name=selected_collection,
                        use_multi_query=True
                    )
                except Exception as e:
                    logger.error(f"Error generating response: {str(e)}")
                    response = "Womp Womp! Error occured"

            # Display the response
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})


def main():
    """Main function to run streamlit app"""
    init_processors()
    init_session_state()
    upload_document()
    display_chat_interface()

if __name__ == "__main__":
    main()
            