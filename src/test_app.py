import json
import os
from typing import List, Dict

# Import required components
from src.document_processing.processor import DocumentProcessor
from src.retrieval.enhanced_retriever import EnhancedRetriever
from src.llm.context_builder import ContextBuilder
from src.llm.handler import LLMHandler
from src.testing.test_generator import TestGenerator
from src.utils.logger import get_logger
from src.config.settings import get_settings

logger = get_logger()
settings = get_settings()

def print_question(question: Dict, index: int = 0):
    """Pretty print a question object"""
    print(f"\n{'='*80}")
    print(f"Question {index+1} ({question.get('strategy', 'unknown strategy')})")
    print(f"{'='*80}")
    print(f"Topic: {question.get('subtopic', 'N/A')}")
    print(f"\nQ: {question.get('question', '')}")
    
    options = question.get('options', [])
    for i, option in enumerate(options):
        correct = option == question.get('correct_answer', '')
        marker = "✓" if correct else " "
        print(f"  {marker} {chr(65+i)}. {option}")
    
    print(f"\nExplanation: {question.get('explanation', '')}")
    
    # Print metadata if available
    metadata = {k: v for k, v in question.items() 
               if k not in ['question', 'options', 'correct_answer', 'explanation', 'id']}
    if metadata:
        print(f"\nMetadata: {json.dumps(metadata)}")

def main():
    """Run the test application"""
    print("\n" + "="*80)
    print("Teaching Assistant - Question Generation Test")
    print("="*80 + "\n")
    
    # Initialize components
    print("Initializing components...")
    doc_processor = DocumentProcessor()
    enhanced_retriever = EnhancedRetriever(document_processor=doc_processor)
    context_builder = ContextBuilder(enhanced_retriever=enhanced_retriever)
    llm_handler = LLMHandler(context_builder=context_builder)
    test_generator = TestGenerator(llm_handler=llm_handler, context_builder=context_builder)
    
    # Get list of available collections
    collections = doc_processor.db_client.list_collections()
    collection_names = [name for name in collections]
    
    print("\nAvailable collections:")
    for i, name in enumerate(collection_names):
        print(f"  {i+1}. {name}")
    
    if not collection_names:
        print("  No collections found. Please index some documents first.")
        return
    
    # Get input from user
    collection_idx = int(input("\nSelect collection number: ")) - 1
    if collection_idx < 0 or collection_idx >= len(collection_names):
        print("Invalid selection")
        return
    
    collection_name = collection_names[collection_idx]
    topic = input("Enter topic to generate questions about: ")
    num_questions = int(input("Number of questions to generate (5-15): ") or "10")
    
    # Generate test
    print(f"\nGenerating {num_questions} MCQ questions on '{topic}'...")
    result = test_generator.generate_mcq_test(
        topic=topic,
        collection_name=collection_name,
        num_questions=num_questions,
        difficulty="medium"
    )
    
    if result["status"] != "success":
        print(f"Error: {result.get('message', 'Unknown error')}")
        return
    
    # Display questions
    test = result["test"]
    questions = test.get("questions", [])
    
    print(f"\nGenerated {len(questions)} questions successfully!")
    
    # Analyze question distribution by strategy
    strategies = {}
    for q in questions:
        strategy = q.get("strategy", "unknown")
        strategies[strategy] = strategies.get(strategy, 0) + 1
    
    print("\nQuestion distribution by strategy:")
    for strategy, count in strategies.items():
        print(f"  {strategy}: {count} questions ({count/len(questions)*100:.1f}%)")
    
    # Display each question
    for i, question in enumerate(questions):
        print_question(question, i)
    
    # Save test to file
    save = input("\nSave test to file? (y/n): ")
    if save.lower() == 'y':
        filename = input("Enter filename (default: test.json): ") or "test.json"
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Test saved to {filename}")

if __name__ == "__main__":
    main()