from pathlib import Path

def clean_text(text: str) -> str:
    """Clean text content for chunking"""
    # Remove excessive whitespaces
    text = ' '.join(text.split()) 
    return text.strip()

def validate_collection_name(name: str):
    """Validate collection name for ChromaDB"""
    import re
    # Check length (3-63 characters)
    if not (3 <= len(name) <= 63):
        return False
    
    # Check pattern (alphanumeric, hyphens, underscores)
    pattern = r'^[a-zA-Z0-9][a-zA-Z0-9_-]*[a-zA-Z0-9]$'
    if not re.match(pattern, name):
        return False
        
    return True
