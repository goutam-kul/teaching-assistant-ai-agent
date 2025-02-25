from typing import List, Dict, Optional
import json
from ollama import Client
from src.llm.prompts import STRICT_EXPLANATION_PROMPT
from src.utils.logger import get_logger
from src.config.settings import get_settings
from src.utils.exceptions import ContextError, LLMError

settings = get_settings()
logger = get_logger()


class LLMHandler:
    """Main class to handle LLM calls"""

    def __init__(self, context_builder):
        """Initialize LLMHandler object"""
        self.client = Client(
            host=settings.OLLAMA_HOST,
        )
        self.model = settings.LLM_MODEL
        self.temperature = settings.LLM_TEMPERATURE
        self.context_builder = context_builder

    def _make_request(
        self, prompt: str, temperature: Optional[float] = None
    ) -> str:
        """Make calls to Ollama"""
        logger.info("Getting response from Ollama")
        try:
            # Get response 
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                stream=False,
                options={
                    "temperature": temperature or self.temperature,
                    "top_p": 0.1,  # reducing the probability distribution
                    "top_k": 10   # reduce creativity and hallucination
                }
            )

            logger.info(f"LLM Response returned with {len(response['response'])} characters")
            return response['response']
        
        except Exception as e:
            logger.error(f"Failed to get response: {str(e)}")
            raise ConnectionError(f"Failed to generate response: {str(e)}")
        
    def explain_topic(
        self,
        topic: str, 
        collection_name: str,
        use_multi_query: bool = True
    ) -> str:
        """Generate explanation for give topic and context"""
        # Get context for the topic
        context = self.context_builder.get_explanation_context(
            topic=topic,
            collection_name=collection_name,
            use_multi_query=use_multi_query
        )
        if not context:
            return "I couldn't find any relevant information about this topic in the given knowledge base."

        # Log the context for debugging
        logger.info(f"explanation context:\n {context}")

        # Format the explanation prompt 
        prompt = STRICT_EXPLANATION_PROMPT.format(topic=topic, context=context)
        logger.info(f"Generated prompt: {prompt}")
        try: 
            # Make request to LLM   
            return self._make_request(
                prompt=prompt,
                temperature=0.1
            )
        except ContextError as e:
            logger.error(f"Context error: {str(e)}")
            return f"I encountered an issue retrieving information: {str(e)}"
        except LLMError as e:
            logger.error(f"LLM error: {str(e)}")
            return f"I encountered an issue generating an explanation: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return f"An unexpected error occurred: {str(e)}"






