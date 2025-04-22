import os
from typing import List, Optional

from adalflow import Sequential
from adalflow.core.db import LocalDB
from adalflow.core.types import Document

from adalflow.utils import get_adalflow_default_root_path

from logging_config import (
    log_info,
    log_success,
    log_warning,
    log_error,
    # log_debug
)


class DatabaseManager:
    """
    Manages the creation, loading, transformation, and persistence of LocalDB instances.
    """

    def __init__(self, repo_name: Optional[str] = None):
        """
        Initialize the database manager.
        
        Args:
            repo_name (str, optional): Name of the repository
        """
        self.db = LocalDB()
        self.source_dir = None
        self.db_path = None
        self.repo_name = repo_name or "repository"
        log_info(f"DatabaseManager initialized for repo: {self.repo_name}")

    def load_or_create_db(self, source_dir: Optional[str] = None) -> bool:
        """
        Load an existing database or create a new one.
        
        Args:
            source_dir (str, optional): Path to the repository
            
        Returns:
            bool: True if the database needs to be transformed, False otherwise
        """
        # Setup paths
        root_path = get_adalflow_default_root_path()
        
        if source_dir and os.path.exists(source_dir):
            # Use the provided source directory directly if it exists
            self.source_dir = source_dir
            log_info(f"Using provided source directory: {self.source_dir}")
        else:
            # Otherwise, construct a path in the .adalflow directory
            self.source_dir = os.path.join(root_path, "repositories", self.repo_name)
            log_info(f"Constructed repository path: {self.source_dir}")
        
        # Database path is always in the .adalflow directory
        self.db_path = os.path.join(root_path, "databases", f"{self.repo_name}.pkl")
        log_info(f"Database path: {self.db_path}")
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Check if the database exists
        if os.path.exists(self.db_path):
            try:
                self.db = LocalDB.load_state(self.db_path)
                docs = self.db.get_transformed_data("split_and_embed")
                if docs:
                    log_success(f"Loaded existing database with {len(docs)} documents")
                    return False  # No need to transform
                else:
                    log_warning("Existing database contains no transformed documents")
                    return True  # Need to transform
            except Exception as e:
                log_error(f"Failed to load existing database: {e}")
                return True  # Need to create and transform
        
        log_info("No existing database found, will need to create new one")
        return True  # Need to create and transform
    
    def transform_documents_and_save(self, documents: List[Document], pipeline: Sequential) -> None:
        """
        Transform documents and save to database.
        
        Args:
            documents (List[Document]): List of documents to transform
            pipeline (Sequential): Transformation pipeline
        """
        log_info(f"Transforming {len(documents)} documents")
        
        # Register the transformer
        self.db.register_transformer(transformer=pipeline, key="split_and_embed")
        
        # Load documents into the database
        self.db.load(documents)
        
        # Transform the documents
        self.db.transform(key="split_and_embed")
        
        # Save the database
        self.db.save_state(filepath=self.db_path)
        
        log_success(f"Transformed and saved {len(documents)} documents to {self.db_path}")
