class TeachingAssistantError(Exception):
    """Base exception class for Teaching Assistant"""
    pass

class DocumentProcessError(TeachingAssistantError):
    """Raised when document processing fails"""
    pass

class LLMError(TeachingAssistantError):
    """Raised when LLM interaction fails"""
    pass

class ContextError(TeachingAssistantError):
    """Raised when context retrieval fails"""
    pass