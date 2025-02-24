from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings

class ChromaDBClient:
    """Singleton ChromaDB client manager"""
    _instance = None

    def __new__(cls, persist_dir: str = "db"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
        return cls._instance

    def get_or_create_collection(self, name: str):
        """Get or create a collection"""
        return self.client.get_or_create_collection(name=name)
    
    def delete_collection(self, name: str):
        """Delete and existing collection using name"""
        try:
            self.client.delete_collection(name=name)
            return True
        except Exception:
            return False
        
    def list_collections(self):
        """Return list of collections"""
        return self.client.list_collections()
        
            
    