# Templates
TEMPLATE = """You are an AI language model assistant. Your task is to generate Five
different versions of the given user question to retrieve relevant documents from a vector
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search.
Provide these alternative questions separated by newlines. Original question: {question}"""
 


# Prompt Template
EXPLAIN_TOPIC_TEMPLATE = """Please explain the concept of {topic} in detail with examples"""

EVALUATE_UNDERSTANDING_TEMPLATE = """Evaluate the following response about {topic} and provide feedback:
User's response: {response}

Please provide:
1. What was correct
2. What needs improvement
3. Suggestions for better understanding
"""


GENERATE_QUESTION_PROMPT = """Generate a multiple choice question about '{topic}' based on this context:

        Context:
        {context}
        
        Create a question that tests understanding of key concepts. Return as JSON with these exact keys:
        - question: The question text
        - options: Array of 4 answer options
        - correct_answer: The full text of the correct option
        - explanation: Why the answer is correct
        
        Example format:
        {{
            "question": "What was the main feature of the Mansabdari system?",
            "options": [
                "A military ranking system",
                "A tax collection system",
                "A religious policy",
                "A trade regulation"
            ],
            "correct_answer": "A military ranking system",
            "explanation": "The Mansabdari system was primarily a military..."
        }}"""



GENERATE_EXPLANATION_PROMPT = """You are a teaching assistant tasked with answering questions based ONLY on the provided context.
Generate a meaningful explanation provided by the given context.

Question: {topic}

Relevant Context: {context}

INSTRUCTIONS:
1. Answer ONLY using information form the context above
2. Summarize the context to create a meaninful explanation 
3. If the context doesn't have relevant information, say "I couldn't find any relevant information about this topic in the given knowledge base."
4. Cite specific parts from the context.
5. Be concise and accurate
6. Do not make up or add information not present in the context.

Your response: 
"""