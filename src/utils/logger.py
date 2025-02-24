import logging

def get_logger():
    """Get logger"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger